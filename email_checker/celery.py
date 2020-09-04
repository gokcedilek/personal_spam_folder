from celery import Celery
app = Celery("email_checker",
             include=["email_checker.tasks"])

app.config_from_object("email_checker.celeryconfig")

app.conf.task_routes = {
    "email_checker.tasks.data_fetching": {
        "queue": "queue1"
    },
    "email_checker.tasks.model_training": {
        "queue": "queue2"
    },
    "email_checker.tasks.plotting": {
        "queue": "queue3"
    }
}

app.conf.beat_schedule = {
    "data_fetching": {
        "task": "email_checker.tasks.data_fetching",
        "schedule": 1 * 30
    },
    "model_training": {
        "task": "email_checker.tasks.model_training",
        "schedule": 1 * 45
    },
    "plotting": {
        "task": "email_checker.tasks.plotting",
        "schedule": 1 * 60
    }
}
