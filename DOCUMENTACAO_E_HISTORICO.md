# 🎙️ Clonador de Voz Moderno (Flow Matching DiT & Voice-to-Voice)

Documento explicativo sobre o funcionamento da aplicação, arquitetura técnica, modalidades disponíveis e o histórico detalhado dos desafios e soluções implementadas.

---

## 📌 1. O que o Aplicativo Faz?

O **Clonador de Voz Moderno** é uma solução de inteligência artificial desenvolvida para clonagem de voz realista e conversão de fala em tempo real, utilizando a arquitetura de **Conditional Flow Matching com Diffusion Transformers (DiT)** e **Vocoder Neural Vocos** a 24kHz, acelerada por hardware na GPU **NVIDIA GeForce RTX 4090**.

A aplicação conta com três módulos principais e uma interface gráfica desktop nativa em **PyQt6 (Dark Theme)**.

### 🌟 Principais Recursos e Modalidades:

```
+-----------------------------------------------------------------------------------+
|                         CLONADOR DE VOZ MODERNO (PyQt6)                          |
+--------------------------+------------------------------+-------------------------+
|   💬 Texto para Voz      |   🔄 Áudio para Áudio        |  🏋️ Treinamento         |
|         (TTS)            |      (Voice-to-Voice)        |     (Fine-Tuning)       |
|                          |                              |                         |
| • Digite qualquer texto  | • Envie áudio gravado/cantado| • Treine vozes próprias |
| • Clone com 3-10s voz    | • Whisper ASR detecta a fala | • Checkpoints .pt       |
| • Português BR Nativo    | • Transforma na voz alvo     | • EMA + Mixed Precision |
+--------------------------+------------------------------+-------------------------+
|              🎭 Troca de Falante (conversa com várias pessoas)                   |
|   pyannote diariza -> você escolhe 1 falante -> só ele é regerado e recolado     |
+-----------------------------------------------------------------------------------+
```

1. **💬 Modo Texto para Voz (TTS Voice Cloning - Zero-Shot)**:
   - Você digita qualquer texto arbitrário.
   - Fornece um pequeno áudio de referência (3 a 10 segundos) da pessoa que deseja clonar.
   - O sistema gera a fala em **Português Brasileiro nativo sem sotaque estrangeiro** em menos de **2 segundos** na RTX 4090.

2. **🔄 Modo Áudio para Áudio (Voice-to-Voice / Voice Conversion)**:
   - Você envia uma gravação de áudio de uma pessoa falando ou cantando. **Um áudio entra, um áudio sai** — não precisa diarizar nada.
   - Também tem o seletor de **Motor**: `Seed-VC` (padrão, quadro a quadro, mesma duração e entoação) ou `F5-TTS` (transcreve e refala).
   - No motor **F5-TTS**, o **Whisper ASR** extrai a mensagem e os tempos de fala na GPU.
   - O motor de Flow Matching sintetiza essa fala com o timbre e as características da **Voz Alvo**.
   - Observação: o ritmo não é copiado do áudio de origem — a duração é estimada pelo tamanho do texto e pela cadência da referência (`duração = ref_audio_len + ref_audio_len / ref_text_len * gen_text_len / speed`).

3. **🎭 Modo Troca de Falante (Conversa com várias pessoas)**:
   - Você envia uma conversa inteira, com duas ou mais pessoas, sem recortar nada à mão.
   - O **pyannote.audio** faz a *diarização*: descobre sozinho quantas pessoas falam e marca quem fala em cada intervalo (`SPEAKER_00`, `SPEAKER_01`, ...).
   - Você escolhe **um** falante; só os trechos dele são regerados com a voz alvo e recolados na linha do tempo original. Os demais continuam com o áudio original, intactos.
   - **Dois motores**, selecionáveis por `--engine`:
     - **`seedvc` (padrão)** — *voice conversion quadro a quadro*. Extrai conteúdo fonético + F0 de cada quadro e ressintetiza com o timbre alvo. **Não existe texto no meio**: a duração é idêntica à do trecho original e a entoação é preservada, então o lip sync se mantém. É o equivalente ao que RVC e So-VITS fazem, mas *zero-shot* (sem treinar por voz).
     - **`f5`** — o caminho antigo, `Whisper → F5-TTS`. Só faz sentido quando você quer **mudar o que é dito**, não só quem diz. A duração e a inflexão são reinventadas.
   - `--fit_mode` só se aplica ao motor `f5`: `stretch` pede a duração exata via `fix_duration`; `pad` preserva o ritmo natural e empurra a linha do tempo.
   - Cada emenda leva crossfade de 15 ms e casamento de RMS com o trecho original, para a troca não saltar ao ouvido.
   - Trechos em que o tempo original não comporta o texto falado são marcados **APERTADO** no relatório (com a razão `natural/slot`), porque saem acelerados ou arrastados.
   - `--seed` fixa a amostragem: sem ela cada rodada sorteia uma nova, então um trecho que saiu ruim costuma melhorar só rodando de novo.

   - Comparação medida neste projeto (florinda ➔ voz do silvio, 7,68 s):

     | motor | erro de duração | similaridade com a voz alvo | correlação de F0 com a fonte |
     |---|---|---|---|
     | Seed-VC (com F0) | **0,1 %** | **0,797** | **0,782** |
     | Seed-VC (sem F0) | 0,1 % | 0,774 | 0,561 |
     | F5-TTS (ASR ➔ TTS) | 5,0 % | 0,740 | **−0,154** |

     A correlação de F0 negativa do F5-TTS mostra que a entoação não tem relação com o original — ela é inventada do zero.

4. **👥 Biblioteca de Personagens**:
   - Relaciona **um ou mais áudios** a um nome de personagem, uma vez só.
   - Depois o personagem aparece num seletor nas abas de TTS, Áudio ➔ Áudio e Troca de Falante, como alternativa a procurar o arquivo toda vez.
   - Com **vários áudios**, eles são emendados numa referência única (até 25 s, com um respiro entre eles) — mais material de voz costuma melhorar a clonagem. O resultado fica em cache e é refeito sozinho quando a lista muda.
   - Guarda também a transcrição opcional da referência, usada pelo motor F5-TTS.
   - O registro fica em `voices/characters.json`, com caminhos relativos à raiz do projeto para o cadastro sobreviver a mover a pasta.
   - Também funciona na CLI: `--character "Silvio"` no lugar de `--ref_audio`, e `--list_characters` para ver o que está salvo.

5. **🎬 Editor de Cena (marcação manual, N falantes)**:
   - Abre o áudio e desenha a **forma de onda**; você arrasta sobre ela para marcar quem fala em cada trecho.
   - Quantos falantes quiser — cada um ganha uma cor, e você escolhe um personagem da biblioteca para cada.
   - Arrastar no vazio cria um trecho; arrastar a borda redimensiona; arrastar o meio move; `Delete` apaga; duplo clique toca só aquele trecho. `Ctrl` + roda do mouse dá zoom, roda sozinha rola.
   - **Uma passada por voz**: os trechos de cada personagem são emendados e convertidos juntos, e depois recolados na linha do tempo. O que não foi marcado fica com o áudio original.
   - **Abre MP4 direto** (também MKV, MOV, AVI, WEBM): a trilha é extraída para um WAV temporário, a onda e a conversão trabalham nele, e o vídeo original fica guardado.
   - Com vídeo aberto, aparece **🎞️ Salvar MP4**: grava o vídeo com a trilha nova. O fluxo de imagem é **copiado sem recodificar** (`-c:v copy`) — rápido e sem perda. Abrindo só áudio, o botão não aparece.
   - **Arrastar e soltar**: jogue o arquivo em cima da onda (ela realça) ou em qualquer campo de caminho das outras abas.
   - A marcação salva em `.json`, para retomar o trabalho depois sem remarcar tudo.
   - Existe porque a diarização automática erra a contagem de falantes em material difícil (ver problema **#20**). Aqui você manda.

6. **🏋️ Modo de Treinamento / Fine-Tuning**:
   - Permite treinar ou refinar modelos com datasets de áudio locais.
   - Suporte a Média Móvel Exponencial de Pesos (**EMA**) para estabilidade e **Mixed Precision (AMP)** para máxima eficiência na GPU.

7. **🖥️ Interface Gráfica Desktop & Inicializador**:
   - Desenvolvida em **PyQt6** com tema escuro profissional.
   - Sete abas: TTS, Áudio ➔ Áudio, **🎭 Troca de Falante**, **👥 Personagens**, **🎬 Editor de Cena**, Treinamento e Status.
   - Todas as abas de conteúdo ficam dentro de uma área rolável: em tela pequena elas rolam em vez de espremer os widgets.
   - Na aba de Troca de Falante: botão **🔍 Detectar**, lista com quanto cada um fala (`SPEAKER_00 — 98.8s em 38 trechos (70.2%)`), botão para **ouvir a amostra** de cada um antes de escolher, seletor de encaixe (sincronia × ritmo natural), seed, e o relatório trecho a trecho com as marcas **APERTADO**.
   - Threads assíncronas em segundo plano (`QThread`) para nunca travar a interface durante a síntese.
   - Player de áudio embutido nativo do Windows e botão "Salvar Como...".
   - Execução direta via **start_gui.bat**.

---

## 🏗️ 2. Arquitetura Técnica

- **Acoustic Backbone**: Diffusion Transformer (**DiT**) de 22 camadas com AdaLN-Zero e RoPE (Rotary Positional Embeddings).
- **Matching Paradigm**: Conditional Flow Matching (**CFM**) com amostragem via Ordinary Differential Equation (**ODE**) solvers (Euler / Midpoint).
- **Language Foundation Model**: `F5-TTS Brazilian Portuguese` (`model_stable.safetensors`), especializado em dicção e fonética brasileira. Carregado com a arquitetura **`F5TTS_Base` (v0)**, que é aquela em que foi treinado — ver problema **#7**.
- **Audio Decoding / Vocoder**: `Vocos` neural vocoder a 24.000 Hz, eliminando artefatos metálicos e robóticos comuns em modelos anteriores (como RVC ou So-VITS).
- **Vídeo**: `ffmpeg` (via `inference/media.py`) para extrair a trilha e remontar o MP4. A imagem nunca é recodificada.
- **Voice Conversion**: `Seed-VC` (`Plachta/Seed-VC` + `BigVGAN v2 44kHz` + `RMVPE` para F0), quadro a quadro e zero-shot. Motor padrão da Troca de Falante — preserva duração e entoação, ao contrário do caminho ASR ➔ TTS.
- **Speaker Diarization**: `pyannote/speaker-diarization-3.1` (via `pyannote.audio` **3.3.2**) para descobrir quantos falantes existem e rotular cada trecho. Requer aceitar os termos no HuggingFace — ver problema **#11**.
- **Speech Recognition (ASR)**: `OpenAI Whisper Large v3 Turbo` (com fallback para `Whisper Small`) em modo long-form (`chunk_length_s=30`) para transcrição no modo Áudio ➔ Áudio.

---

## 🛠️ 3. Histórico de Problemas Enfrentados e Soluções

Abaixo estão detalhados todos os problemas encontrados durante o desenvolvimento e as soluções técnicas adotadas:

| # | Problema Identificado | Causa Raiz | Solução Aplicada | Status |
|---|---|---|---|:---:|
| **1** | **Ruídos e estática nas sínteses iniciais** | Treinar um modelo DiT do zero com poucos segundos de dados sintéticos gera pesos aleatórios que produzem apenas estática. | Integração do Foundation Model SOTA pré-treinado em milhares de horas de fala (`F5-TTS`) + `Vocos 24kHz`. | ✅ Resolvido |
| **2** | **Falha de DLL no Windows (`torchcodec [WinError 127]`)** | O pacote opcional `torchcodec` continha binários com procedimentos ausentes no Windows, quebrando o pipeline do Transformers. | Remoção do `torchcodec` e uso de pipeline direto via `soundfile` e `torchaudio` nativo. | ✅ Resolvido |
| **3** | **Sotaque de inglês estadunidense ao falar português** | O modelo base internacional utilizava fonemas treinados predominantemente na língua inglesa. | Integração do modelo especialista em Português Brasileiro (`traderpedroso/F5-TTS-BRAZILIAN-PORTUGUESE`). | ✅ Resolvido |
| **4** | **Necessidade de Conversão Áudio ➔ Áudio (Voice Conversion)** | O sistema inicial operava exclusivamente por texto digitado. | Criação de uma aba e pipeline dedicados de Voice-to-Voice com extração de fala via Whisper ASR na GPU. | ✅ Resolvido |
| **5** | **Erro de canais estéreo no Whisper (`shape (460800, 2)`)** | Arquivos de áudio estéreo (2 canais) eram passados diretamente para o Whisper, que espera 1 canal mono. | Implementação de downmixing automático (`audio.mean(axis=-1)`) e conversão para `float32`. | ✅ Resolvido |
| **6** | **Erro de Encoding no Console Windows (`cp1252 charmap \u2794`)** | Caracteres de seta Unicode (`➔`) causavam erro ao imprimir no console padrão do Windows. | Substituição de caracteres especiais por padrão ASCII seguro (`->`). | ✅ Resolvido |
| **7** | **Fala sai como fonemas sem nexo, com o timbre correto** | O checkpoint PT-BR (`model_stable.safetensors`) é um fine-tune do **F5TTS_Base (v0)**, mas era carregado como `F5TTS_v1_Base`. As duas arquiteturas têm shapes idênticos, então o `load_state_dict` passava em silêncio — porém o v1 usa `pe_attn_head=None` (RoPE em todas as cabeças) e `text_mask_padding=True`, enquanto o v0 foi treinado com `pe_attn_head=1` e `text_mask_padding=False`. O alinhamento texto/áudio quebrava: o timbre saía certo e a fala virava fonética aleatória. | Arquitetura amarrada ao checkpoint em `cloner.py` (`PT_BR_ARCH = "F5TTS_Base"`), com parâmetro `model_arch` para override manual. | ✅ Resolvido |
| **8** | **`ref_text` descrevendo mais áudio do que o modelo escuta** | O F5-TTS recorta a referência em ~12s internamente, mas usa o `ref_text` recebido por inteiro. Uma referência de 19s era transcrita completa (274 bytes) e casada com apenas 11,1s de áudio — prompt desalinhado e duração mal estimada. | Método `_prepare_reference()`: recorta primeiro (via `preprocess_ref_audio_text`) e transcreve **o recorte**, garantindo que texto e áudio descrevam o mesmo trecho. Avisa quando uma transcrição manual deixa de valer por causa do corte. | ✅ Resolvido |
| **9** | **Transcrição imprecisa e truncada em 30s no modo Áudio -> Áudio** | O `whisper-small` errava palavras ("vem da piaidade" no lugar de "vem tapear a idade") e o pipeline sem `chunk_length_s` descarta tudo depois de 30s. No modo Áudio -> Áudio a transcrição **é** o texto falado, então o erro ia direto para o áudio final. | Migração para `whisper-large-v3-turbo` com `chunk_length_s=30` (long-form) e fallback automático para `whisper-small`. | ✅ Resolvido |
| **11** | **`pip install pyannote.audio` quebraria todo o projeto** | A versao 4.x exige `torch>=2.8` e o pip resolveria para `torch 2.13.0` **da PyPI, sem CUDA** (adeus RTX 4090), mais `torchaudio 2.11`, `numpy 2.x` e — pior — `torchcodec 0.16`, exatamente o pacote removido no problema **#2** por causa do `WinError 127`. | Fixado `pyannote.audio==3.3.2` com `numpy<2`: resolve sem tocar em torch, torchaudio nem torchcodec. Comando: `pip install "pyannote.audio==3.3.2" "numpy<2"`. | ✅ Resolvido |
| **12** | **`hf_hub_download() got an unexpected keyword argument 'use_auth_token'`** | O `pyannote.audio` 3.3.2 ainda chama `hf_hub_download(use_auth_token=...)`, argumento removido no `huggingface_hub` 1.x. Nao dava para voltar o hub (o `transformers` 5.x exige `huggingface-hub>=1.5`) nem para subir o pyannote (a 4.x arrasta torch sem CUDA e torchcodec). | Shim `_patch_hf_hub_compat()` em `diarizer.py`: traduz `use_auth_token` para o `token` atual nos tres modulos do pyannote que ainda o usam. | ✅ Resolvido |
| **13** | **`UnpicklingError: Weights only load failed` ao carregar a diarizacao** | O torch 2.6 passou a usar `weights_only=True` por padrao em `torch.load`, e o checkpoint do `pyannote/segmentation-3.0` guarda objetos fora da lista branca. | `_patch_torch_load_compat()` libera apenas as 4 classes que esse checkpoint usa (`TorchVersion`, `Specifications`, `Problem`, `Resolution`), em vez de desligar a verificacao com `weights_only=False` — que executaria pickle arbitrario. | ✅ Resolvido |
| **14** | **Conversa encurtava e os outros falantes saiam de sincronia** | O crossfade de entrada consumia `fade` amostras da cabeca de cada trecho gerado, mas o trecho tinha exatamente o tamanho do slot. Cada emenda encurtava a saida em 15 ms e o erro acumulava: 18 trocas = 0,27 s de deriva, o bastante para desalinhar o audio dos outros falantes. | O trecho passa a ser gerado com `slot + fade` amostras, para sobrar exatamente `slot` depois da sobreposicao. Verificado: 180,000 s -> 180,000 s e `max|diff| = 0` nas janelas dos outros falantes. | ✅ Resolvido |
| **15** | **Interjeicoes curtas saiam com quase 2 s** | O F5-TTS forca `local_speed = 0.3` para texto com menos de 10 bytes, entao um "Nao sei." de 0,46 s virava 1,88 s de audio — e o encaixe posterior comprimia 4x. | No modo `stretch`, `fix_duration` e passado ao F5: ele tem prioridade sobre a estimativa de duracao **e** sobre o hack de velocidade, gerando ja no tempo do slot. Os casos que ainda nao cabem sao marcados APERTADO. | ✅ Resolvido |
| **16** | **Widgets esmagados na aba de Troca de Falante** | A aba tem cinco grupos e exigia **854 px** de altura, contra **625 px** uteis na janela — o Qt espreme os widgets em vez de cortar, deixando tudo ilegivel. Depois, ao envolver o conteudo numa area rolavel com `ScrollBarAlwaysOff`, surgiu o problema oposto: a largura minima era **1537 px** e o conteudo ficava **cortado** ate em 1280 px, porque `QLabel` e `QComboBox` exigem por padrao a largura do texto inteiro. | `QScrollArea` com `setWidgetResizable(True)` e barras `AsNeeded` (nada fica inalcancavel); layout compactado (862 -> 729 px de altura); `setMinimumContentsLength` nos combos, `QSizePolicy.Ignored` no label de resultado e titulos/rotulos curtos (1537 -> 836 px de largura); janela abre em 1020x900 com `fit_to_screen()` reduzindo em monitores menores. Verificado em 760x600, 880x760, 1020x900 e 1280x1000. | ✅ Resolvido |
| **17** | **Audio convertido menor, sem lip sync e com fonemas faltando** | O pipeline era `Whisper -> F5-TTS`: audio vira TEXTO e depois audio de novo. Tudo que nao e texto (duracao, curva de pitch, enfase, pausas) e descartado no caminho e reinventado pelo TTS. Palavra que o ASR nao ouviu simplesmente nao existe na saida. Medido: erro de duracao 5,0% e correlacao de F0 com a fonte **-0,154** (nenhuma relacao). Nenhum ajuste de `fit_mode`, seed ou passos ODE resolve — e limitacao de arquitetura. | Integracao do **Seed-VC** como motor padrao (`--engine seedvc`): voice conversion quadro a quadro, zero-shot, sem texto no meio. Medido: erro de duracao **0,1%**, correlacao de F0 **0,782**, similaridade com a voz alvo 0,797 (contra 0,740 do F5). O caminho F5 continua disponivel para quando o objetivo e mudar o que e dito. | ✅ Resolvido |
| **18** | **`BigVGAN._from_pretrained() missing 2 required keyword-only arguments`** | O BigVGAN embutido no seed-vc declara `proxies` e `resume_download` como obrigatorios, mas o `huggingface_hub` 1.x parou de passa-los — e ainda os repassa ao proprio `hf_hub_download`, que tambem nao os aceita mais. Mesma raiz do problema **#12**, em outra biblioteca. | Os remendos foram centralizados em `inference/hf_compat.py` (`patch_hf_hub_download`, `patch_bigvgan`, `patch_torch_load`), usado pelo diarizer e pelo motor de VC. Traduz a API antiga em vez de mudar versao de pacote. | ✅ Resolvido |
| **19** | **`--num_speakers` parecia nao funcionar na GUI** | As setas so mudam o numero; nada e recalculado ate clicar em **🔍 Detectar** de novo. Nao havia nenhuma indicacao disso na tela. | A aba passa a mostrar o estado: apos detectar informa se a contagem foi automatica ou forcada, e ao mexer nas setas exibe **"⚠️ Contagem alterada — clique em 🔍 Detectar para refazer a analise"**. Verificado que forcar a contagem funciona: com 4, o pyannote divide o SPEAKER_02 em dois. | ✅ Resolvido |
| **20** | **Diarizacao automatica errando a contagem de falantes** | Numa esquete com Chaves, Quico e Seu Madruga o `pyannote` detectou 2 vozes em vez de 3. Vozes parecidas, muita sobreposicao e fundo musical confundem o modelo — e forcar `num_speakers` nem sempre acerta as fronteiras. O caminho era converter os tres por completo e depois cortar num editor externo. | Nova aba **Editor de Cena**: forma de onda com regioes marcadas a mao, N falantes, um personagem para cada, e a mixagem sai pronta numa passada. Backend em `convert_segments()`, que generaliza `convert_speaker()` para varios falantes com trechos vindos de fora em vez da diarizacao. | ✅ Resolvido |
| **21** | **Widget de forma de onda sem dependencia nova** | Plotar a onda pediria `pyqtgraph` ou `matplotlib`; o primeiro e mais uma dependencia para um uso so, e o segundo tem interacao ruim para arrastar bordas de regiao. | `gui_waveform.py`: `QWidget` proprio que desenha no `paintEvent`. Um resumo de picos a 200 baldes/segundo e agregado por coluna de pixel com `np.minimum.reduceat`, o que mantem o desenho leve mesmo num arquivo de 20 minutos, e a interacao (criar, mover, redimensionar, zoom) fica sob controle total. | ✅ Resolvido |
| **22** | **Vaivem para trabalhar com video** | O material de origem e sempre MP4, mas o app so abria audio: era preciso extrair a trilha a mao antes e casar o audio novo com o video depois, num editor. | `inference/media.py` com o ffmpeg (ja instalado na maquina, sem dependencia nova). Abrir MP4 extrai a trilha para um WAV temporario; o botao **Salvar MP4** remonta o video com o audio novo usando `-c:v copy`, que copia o fluxo de imagem sem recodificar. Verificado: `codec_name`, resolucao e `nb_frames` identicos entre entrada e saida, e a trilha do MP4 final bate com a mix (correlacao +1.000) e nao com o original (-0.009). | ✅ Resolvido |
| **10** | **Referências de demo em `data/demo_speaker/` não contêm fala** | Os quatro `.wav` (todos com exatos 3,50s) são tons sintéticos harmônicos (~150/300/450 Hz) da fase inicial do projeto, e os `.txt` irmãos declaram 90 bytes de texto que o áudio nunca fala. Qualquer teste com eles produz resultado ruim mesmo com o pipeline correto. | Usar referências de fala real (ex.: `dataset/silvio/silvio.mp3`) para avaliar qualidade. Pendente substituir os arquivos de demo. | ⚠️ Conhecido |

---

## 🚀 4. Como Executar

### 1. Pela Interface Desktop (Recomendado):
Basta dar dois cliques no arquivo:
👉 **start_gui.bat**

Para trocar a voz de um personagem: aba **🎭 Troca de Falante** ➔ escolha a conversa ➔ **🔍 Detectar** ➔ ouça as amostras para saber quem é quem ➔ selecione o falante e a voz nova ➔ **Trocar a Voz do Falante Selecionado**. O resultado vai para `swap_output.wav` e as amostras para a pasta `speaker_samples/`.

### 2. Por Linha de Comando (CLI):

- **Para Texto ➔ Voz (TTS)**:
  ```bash
  .\venv\Scripts\python.exe inference/infer.py --ref_audio "data/demo_speaker/audio_01.wav" --text "Olá, este é um teste de voz clonada." --output "resultado.wav"
  ```

- **Usando um personagem salvo em vez do caminho do arquivo**:
  ```bash
  .\venv\Scripts\python.exe inference/infer.py --list_characters
  .\venv\Scripts\python.exe inference/infer.py --character "Silvio" --text "Olá, tudo bem?" --output "saida.wav"
  ```

- **Para Áudio ➔ Áudio (Voice-to-Voice)**:
  ```bash
  .\venv\Scripts\python.exe inference/infer.py --source_audio "minha_fala.wav" --ref_audio "voz_alvo.wav" --output "convertido.wav"
  ```

- **Para trocar a voz de UM personagem numa conversa**:

  Primeiro descubra quem é quem (salva uma amostra de cada falante para você ouvir):
  ```bash
  .\venv\Scripts\python.exe inference/infer.py --source_audio "conversa.wav" --list_speakers
  ```

  Depois troque só o falante escolhido (use `auto` para pegar quem mais fala):
  ```bash
  .\venv\Scripts\python.exe inference/infer.py --source_audio "conversa.wav" --speaker SPEAKER_01 --ref_audio "voz_alvo.wav" --output "conversa_trocada.wav"
  ```

  Isso usa o **Seed-VC** por padrão: mesma duração, mesma inflexão, lip sync preservado.

  Opções úteis: `--num_speakers 2` (força a quantidade), `--max_speakers 4`, `--diffusion_steps 35` (mais qualidade, mais lento), `--no_f0` (se o resultado com F0 sair instável).

  Para **mudar o que é dito** em vez de só quem diz, troque o motor:
  ```bash
  .\venv\Scripts\python.exe inference/infer.py --source_audio "conversa.wav" --speaker SPEAKER_01 --ref_audio "voz_alvo.wav" --engine f5 --fit_mode pad --output "saida.wav"
  ```

### 3. Pré-requisito para trabalhar com vídeo

O **ffmpeg** precisa estar acessível para abrir MP4 e gravar o vídeo de saída. Baixe de
https://www.gyan.dev/ffmpeg/builds/ e deixe o `ffmpeg.exe` no PATH, ou instale o pacote que
já traz o binário:

```bash
pip install imageio-ffmpeg
```

O app procura o ffmpeg no PATH, nos caminhos usuais do Windows (`C:\Program Files\FFMPEG\bin`) e,
por último, no `imageio-ffmpeg`. Sem ele, as abas de áudio seguem funcionando normalmente.

### 4. Pré-requisito da diarização (uma vez por conta)

Os pesos do pyannote são *gated*. Aceite os termos nos dois repositórios e faça login:

- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

```bash
huggingface-cli login
```

> ⚠️ Instale **sempre** com a versão fixada: `pip install "pyannote.audio==3.3.2" "numpy<2"`.
> A 4.x troca o `torch` por um build sem CUDA e reintroduz o `torchcodec` (ver problema **#11**).

> ℹ️ `pip check` reclama que `f5-tts requires torchcodec, which is not installed`.
> É esperado e deve continuar assim — o `torchcodec` foi removido de propósito no problema **#2**;
> o projeto usa `soundfile` e `torchaudio` no lugar.
