# Начало разработки

Для начала создайте свою папку и склонируйте репозиторий туда.

Для того, чтобы из сервера не выбрасывало из-за 2-3 минут бездействия, можно прописать в `config` файле папки `~/.ssh` следующее:
```
TCPKeepAlive yes
ServerAliveInterval 30
```

## Создание окружения

Можно использовать Python из `/home/igorl/music_transformer/env/bin` (где стоят все необходимые пакеты), но лучше создать своё окружение.

Инструкция ниже представлена для директории `/home/<some user name>`

1. Создаём копию `requirements.txt` с необходимыми пакетами: `/home/igorl/music_transformer/env/bin/pip freeze > /home/<some user name>/requirements.txt`
2. Создаём окружение: `/home/igorl/music_transformer/env/bin/python3  -m venv <some environment name>`
3. Активируем окружение: `source <some environment name>/bin/activate`
4. Устанавливаем пакеты: `pip install -r /home/<some user name>/requirements.txt`
5. Для завершения работы окружения: `deactivate`

# Обучение

Перед тем как запустить обучение, нужно предобработать датасет. Сейчас обучение ведется в music_transformer/MusicTransformer-Pytorch и все команды представлены для этой директории, в вашем случае поменяйте пути в соответствии с расположением вашей копии.

## Предобработка датасета:

Поиск мажорностей для каждого токена:

1. Кладем нужные midi файлы в ``music_transformer/MusicTransformer-Pytorch/dataset/e_piano/custom_midis``
2. ``cd music_transformer/MusicTransformer-Pytorch``
3. Если работаете в общем датасете, сохраните копию папки ``tension_out`` куда-нибудь, чтобы ее можно было восстановить
4. Запускаем ``python3 /home/igorl/music_transformer/midi-miner/tension_calculation.py -i dataset/e_piano/custom_midis -o dataset/e_piano/tension_out``

Чтобы подготовить midi файлы, запускаем ``python3 preprocess_midi.py <path to dataset> -output_dir <path to output folder>``. Если не указывать ``-output_dir``, файлы будут сохранены в ``dataset/e_piano``

После этого подготовленный датасет будет лежать в указанной output директории в папках ``train, test, val``

Сейчас есть готовые датасеты ``/data/datasets/e_piano_big`` и ``/data/datasets/e_piano_small``

## Запуск обучения:
```
cd music_transformer
sudo CUDA_VISIBLE_DEVICES=0,1 /home/igorl/music_transformer/env/bin/python3 train.py -path_to_params parameters.json -path_to_model_params model/model_params.json
```
Параметры можно менять под себя, есть еще дополнительные, которые управляют размером модели, их можно посмотреть с помощью ``python3 train.py --help `` или в файле ``parameters.json`` и там же поменять. Также обратите внимание на переменную ``CUDA_VISIBLE_DEVICES``, в нее нужно передавать номера гпу, которые в данный момент свободны. Посмотреть какие свободны можно командой ``nvidia-smi``
По умолчанию используется датасет maestro, но он нужен скорее чтоюы проверить модель. Чтобы полноценно обучать, рекомендуется использовать один из ``/data/datasets/e_piano_small`` на 50 эпохах или ``e_piano_big`` на 20.

## Генерация примеров:
```
sudo /home/igorl/music_transformer/env/bin/python3 -m generation.generate_sample
```

Генерация одного примера. Параметры настраиваются в ``generate_sample_parameters.json``

Или

```
sudo /home/igorl/music_transformer/env/bin/python3 -m generation.evaluate_samples
```
Генерация примеров на основе семплов из папки. Параметры настраиваются в ``evaluate_samples_parameters.json``

## Установка с 0 и генерация
```
python3 -m venv env
env/bin/pip install -r requirements.txt
```
Change path to model in generation.gen.sh
```
generation/gen.sh <tokens length> <sentiment int> <genre int> <out dir> <out name> <optional seed>
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
