## Подготовка к запуску

1. Запуск парсера и бд

Чтобы запустить парсер и базы данных можно следовать инструкции, описанной в [patent-parser](https://github.com/iamsuchafitta/patent-parser/tree/243d530f77947e85cbcd63a5788a87d6bb7b1fca)

3. Выполнить миграции для бэкенда

Для этого необходимо установить библиотеку clickhouse_connect:

```bash
pip install clickhouse_connect
```

И выполнить код в [clickhouse_migrations](https://github.com/DelicaLib/patents-analyze/tree/main/src/RuPatents_rel/clickhouse_migrations)

```bash
python main.py
```

## Запуск с помощью Docker (Рекомендовано)

4. Запуск бэкенда `http://localhost:9090` с пресобранного Image:

```bash
docker-compose up
```

5. Проверка работоспособности сервера

```bash
$ curl http://localhost:9090/
{"status":"UP"}
```

6. Подключение Label Studio к бэкенду, если Label Studio находится на той же машине: перейдите в проект `Settings -> Machine Learning -> Add Model` и укажите `http://localhost:9090` как URL.


## Сборка из источников (Продвинутый)

Чтобы собрать бэкенд fиз источников, вы можете склонировать репозиторий и собрать docker image:

```bash
docker-compose build
```

## Запуск без Docker (Продвинутый)

Чтобы запустить бэкенд без Docker, необходимо склонировать репозиторий и установить все зависимости:

```bash
python -m venv ml-backend
source ml-backend/bin/activate
pip install -r requirements.txt
```

Затем можно запустить сам бэкенд:

```bash
python main.py
```

# Конфигурация
Параметры, которые можно выставить в `docker-compose.yml` перед запуском контейнера.


Доступны следующие параметры:
- `LOG_LEVEL` - установить уровень логирования
- `WORKERS` - установить количество workers
- `THREADS` - установить количество потоков
 