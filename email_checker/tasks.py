import os
import redis
from email_checker.celery import app
from email_checker.api_task1 import setup_email, setup_db, fetch_emails, process_emails, process_first_fetch, process_next_fetches
from email_checker.api_task2 import retrieve_emails, hash_encode, fit_clf, get_cv_stats, get_pred_stats, save_stats
from email_checker.api_task3 import plot_stats


# connect to redis
redis_server = redis.Redis("localhost")

# connect to the database
cursor = setup_db(os.environ.get("db_name"), os.environ.get(
    "db_user"), "", "127.0.0.1", "5432")


@app.task
def data_fetching():
    # define redis keys if not defined
    if not (redis_server.exists("next_uid")):
        redis_server.set("next_uid", 1)

    # (re)connect to the mail server everytime the task is executed, to prevent connection timing out
    mail = setup_email(os.environ.get("email_address"),
                       os.environ.get("email_pwd"), "imap.gmail.com")

    # get the uid of the last processed email
    last_uid = int(redis_server.get("next_uid"))
    read_uid = last_uid
    unread_uid = last_uid

    # fetch the emails since last_uid
    read, read_uid = fetch_emails('SEEN', last_uid, read_uid, mail)
    unread, unread_uid = fetch_emails('UNSEEN', last_uid, unread_uid, mail)

    # data preprocessing before saving the emails
    read = process_emails(read)
    unread = process_emails(unread)

    # process the emails
    if last_uid == 1:
        # save the emails to training & test data tables
        process_first_fetch(read, unread, cursor)
    else:
        # save the emails to training data table
        process_next_fetches(read, unread, cursor)

    # store the next_uid {uid of the last fetched email} back
    next_uid = read_uid if read_uid > unread_uid else unread_uid
    # only change the value in redis *if* there are new emails since the last time fetched, otherwise we should NOT increment the next uid value, so that we will continue fetching from the right email the next time!
    if(next_uid != last_uid):
        redis_server.set("next_uid", next_uid)


@app.task
def model_training():
    # define redis keys if not defined
    if not (redis_server.exists("num_trained")):
        redis_server.set("num_trained", 0)

    # get the number of emails used for training from the last execution of this task
    num_trained = int(redis_server.get("num_trained"))

    # get the training & test emails, and the number of emails available for training
    training, training_count = retrieve_emails(cursor, "emails")
    test, _ = retrieve_emails(cursor, "emails_test")

    # if there is new training data available, (re)train
    if num_trained < training_count:
        # hash encode training & test
        train_features, train_label, test_features, test_label = hash_encode(
            training, test)
        # model training
        clf = fit_clf(train_features, train_label)
        # model evaluation
        training_dict = get_cv_stats(clf)
        test_dict = get_pred_stats(clf, test_features, test_label)
        # save stats to csv
        save_stats('training.csv', 'a+', training_dict)
        save_stats('test.csv', 'a+', test_dict)
        # update trained count
        redis_server.set("num_trained", training_count)


@app.task
def plotting():
    # create plots for the training & test metrics
    plot_stats('training.csv', 'test.csv')
