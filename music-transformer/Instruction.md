# Начало разработки (на сервере @iglab)

Для начала создайте своего пользователя и склонируйте репозиторий.

Для того, чтобы из сервера не выбрасывало из-за 2-3 минут бездействия, можно прописать в `config` файле папки `~/.ssh` следующее:
```
TCPKeepAlive yes
ServerAliveInterval 30
```

# Создание окружения (на примере conda)

```bash
cd music-transformer
conda create --name music_transformer python=3.8.5
conda activate music_transformer
pip install -r requirements.txt
```

# Обучение

## Датасет

При обучении модели, как правило, используется объединение датасетов midi-композиций [ADL Piano Midi](https://paperswithcode.com/dataset/adl-piano-midi) и [GiantMIDI-Piano](https://paperswithcode.com/dataset/giantmidi-piano), который (из исторических соображений) именуется как `epiano_small`.

Для запуска обучения необходим датасет, который токенизирован способом, описанным в статье `MusicTransformer`. При работе на сервере @iglab этот датасет лежит в `/data/datasets/e_piano_small_kek`.

На другой машине его можно получить двумя способами:

### Копирование датасета с сервера @iglab (простой способ)

Для датасета `epiano_small` есть предобработанная, токенизированная версия на сервере @iglab, её можно просто скопировать на необходимую машину.

1. Скачиваем датасет через scp, пусть в папку `/home/user/datasets`: 
    ```bash
    scp -r /data/datasets/e_piano_small_kek <user_login>@<host_ip>:/home/user/datasets
    ```
2. Подменяем пути родительской папки `e_piano_small_kek` в файлах `global_features.json`, в команде `sed` в пути родительской папки экранируем /:
    ```bash
    for folder in "train" "test" "val"; do sed -i 's/\/data\/datasets/\/home\/user\/datasets/g' /home/user/datasets/e_piano_small_kek/$folder/global_features.json; done
    ```

### Предобработка датасета

Пусть есть папка с midi-файлами, тогда можно запустить предобработку датасета через команду:

```bash
python preprocessing/preprocess_midi.py
```

При этом параметры предобработки задаются в `preprocessing/preprocessing_parameters.json`, их описание лежит в `preprocessing/parameters_description.md`.

// по вопросам предобработки датасета можно писать @black_chick

## Меняем конфиги под себя

Есть несколько конфигов/файлов, которые полезно поменять перед запуском обучения:

* `dataset/dataset_parameters.json` -- отвечает за параметры датасетов для обучения/валидации/теста. Описание параметров лежит в `dataset/dataset_parameters_description.md`. **Параметром `dataset_root` задаётся путь root'а датасета, его нужно поменять в соответствии с его расположением**.
* `parameters.json` -- отвечает за общие настройки обучения. Описание параметров лежит в `Parameters.md`. Что полезно поменять:
	* `output_dir` -- путь до директории, куда сохранятся веса модели и метрики обучения; в `output_dir/results` хранятся веса для лучшей эпохи по тестовому лоссу/по тестовому accuracy и "поэпохная" статистика значений лосса, метрик; в `output_dir/weights` лежат веса модели, сохраненные в конце каждой из эпох.
	* `batch_size` -- суммарный размер батча, из-за применения `torch.nn.DataParallel` в коде обучения при использовании более одного GPU на одном GPU будет использован размер батча, равный `batch_size` / число использованных для обучения GPU. Более подробно об этом смотри [тут](https://discuss.pytorch.org/t/do-dataparallel-and-distributeddataparallel-affect-the-batch-size-and-gpu-memory-consumption/97194).
	* `epochs` -- число эпох для обучения.
	* `wandb_project_name` - название проекта в WANDB, где сохранится текущий запуск (run) обучения. Проекты являются объединением нескольких запусков. Название специфичного запуска можно задавать из кода: для этого необходимо поменять [эту](https://github.com/glinkamusic/music-transformer/blob/07bc06c1b0d8aaceaccd006fff6bfb81898e1db8/train.py#L96) строчку, альтернативно, название run'а можно поменять в web-интерфейсе WANDB.
* `model/model_params.json` -- отвечает за конфигурацию обучающейся модели: кол-во слоёв, размеры промежуточных представлений и т.п.
* `utilities/constants.py` -- разные константы для параметров начального learning rate, параметров оптимизатора, константы для границ событий разного типа, использующихся токенизатором `MusicTransformer` и т.п.


## Запуск обучения

Обычно обучение запускается в отдельном консольном окне, независящем от Вашего присутствия на (удаленной) машине; его можно получить, например, через утилиту `tmux`,  [ссылка на гайд по tmux](https://habr.com/ru/articles/327630/):

```bash
tmux new -s <название сессии>
```

Выйти из сессии (**при этом запущенный там процесс не прекратится!**): комбинация клавиш `CTRL+B D` (комбинация для Linux, а вообще платформозависимо)

Зайти в ранее созданную сессию `<название сессии>`:

```bash
tmux attach -t <название сессии>
```

В открывшемся окне нужно активировать окружение заново, после чего прописать команду для запуска обучения:

```bash
CUDA_VISIBLE_DEVICES=<номера использующихся GPU через запятую> python train.py -path_to_params parameters.json -path_to_model_params model/model_params.json
```

### Про GPU

Про использование GPU можно узнать через команду `nvidia-smi` -- ориентируемся на загруженнность памяти карточек. Альтернативно, информацию про GPU можно узнать через `Python`-пакет `nvitop` (ставится через `pip install nvitop`, запускается как `nvitop -m`), он предоставляет более детальную информацию о процессах, занимающих GPU, например, имя пользователя, запустившего процесс, суммарное время работы процесса.

**Важно: при работе на сервере @iglab не занимать 0-ую и 1-ую GPU, они используются для инференса продовых моделей!**

### Про WANDB

При первом запуске WANDB потребует указать, использовать ли его для логирования, и если да, то нужно ли создать для этого аккаунт или использовать существующий (для этого нужно указать API ключ). Обычная практика: результаты хранятся на аккаунте проекта (реквизиты к нему спрашивать у @iglab), после захода на него API ключ можно найти в User Settings > API Keys. Артефакты от WANDB по умолчанию сохранятся в `/data/wandb`, при отсутствии `/data` -- в `/tmp/wandb`. Эту директорию можно поменять на [этой](https://github.com/glinkamusic/music-transformer/blob/07bc06c1b0d8aaceaccd006fff6bfb81898e1db8/train.py#L102) строчке.

Если всё ок, то появится сообщение об evaluation'е на 0-ой эпохе (когда модель случайно иницилизирована)...
```
Baseline model evaluation (Epoch 0):
```

... а спустя время должна появиться статистика про лосс на батчах следующего вида:
```
=========================
Epoch 1  Batch 211 / 4281
LR: 3.0096213446247416e-05
Train loss: 4.825368881225586

Time (s): 0.5759422779083252
=========================
```

## Генерация примеров

**Важно: для генерации моделью необходимо иметь конфиги `dataset/dataset_parameters.json`, `model/model_params.json`, которые использовались при обучении этой самой модели**

### Генерация через bash-скрипт

```bash
python3 -m generation.gen_script <путь до .pickle файла с весами модели> <длина генерируемой композиции> <индекс сантимента, см. utilities/constants.py> <индекс жанра, см. utilities/constants.py> <директория со сгенерированной композиицией> <название сгенерированной композиции> <seed для сэмплирования>
```

### Генерация через json-конфиг

Более детальная генерация, с большим количеством параметров. Параметры настраиваются в `generation/generate_sample_parameters.json`, их описание есть в `generation/parameters_description.md`
```bash
python3 -m generation.generate_sample
```

# Проверка на codestyle

Проверка осуществляется [плагином](https://github.com/wemake-services/wemake-python-styleguide) `wemake-python-styleguide` для линтера `flake8`.

Поставить его можно так:
```
pip install -r requirements.txt
```

Для локального использования *внутри репозитория* (внутри -- потому что линтеру необходим конфигурационный файл `setup.cfg`, который лежит в корне репозитория):
```
flake8 file_name.py
```

Линтер предоставляет вывод обнаружившихся ошибок с их кратким описанием и местами в коде, где они возникли. Полный список ошибок с более детальным описанием лежит [тут](https://wemake-python-styleguide.readthedocs.io/en/latest/pages/usage/violations/index.html).

### Что делать, если проверки линтера не проходят?

0. Попробовать их исправить :)
1. Если ошибка - "единичный случай", то проверку линтера для неё можно отключить добавлением комментария `# noqa: <error_code>` на соответствующей строке, например,
    ```
    example = lambda: 'example'  # noqa: E731
    ```
   Подробнее об игнорировании ошибок `flake8` написано [тут](https://flake8.pycqa.org/en/latest/user/violations.html).
2. Если одна и та же ошибка возникает несколько раз на уровне файла, то можно в конфиге `setup.cfg` прописать её игнорирование на уровне этого файла в секции `per-file-ignores`, с (желательным) описанием того, зачем её *хочется* проигнорировать:
    ```
    # setup.cfg
    [flake8]
    ...
    ...
    
    per-file-ignores =
          # These function names are part of 3d party API:
          wemake_python_styleguide/visitors/ast/*.py: N802
          # These modules should contain a lot of classes:
          wemake_python_styleguide/violations/*.py: WPS202
    ```
3. Если ошибка возникает на уровне нескольких файлов, то можно воспользоваться предыдущим пунктом (указав файлы через запятую) или отключить проверку этого чека, указав его в секции `ignore` в `setup.cfg`:
    ```
    # setup.cfg
    [flake8]
    ...
    ...
    
    ignore = D100,D104,N812
    ```
    Подробнее об игнорировании ошибок `wemake-python-styleguide` написано [тут](https://wemake-python-styleguide.readthedocs.io/en/latest/pages/usage/configuration.html#).
