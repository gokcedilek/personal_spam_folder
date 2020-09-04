import psycopg2
from psycopg2 import OperationalError
import imaplib
import email
import string
import re
import math
import random
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# define constants
TEST_RATIO = 0.2


def setup_email(user, password, imap_url):
    # SSL: encrypted connection
    connection = imaplib.IMAP4_SSL(imap_url)
    connection.login(user, password)
    connection.select('INBOX', readonly=True)
    return connection


def setup_db(db_name, db_user, db_password, db_host, db_port):
    connection = None
    try:
        connection = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        print("Connected to PostgreSQL!")
    except OperationalError as e:
        print(f'Exception: {e}\n')
    if(connection):
        connection.autocommit = True
        cursor = connection.cursor()
        # create the training table
        cursor.execute(
            "create table if not exists emails (id serial primary key, sender text not null, subject text, label integer not null)")
        # create the testing table
        cursor.execute(
            "create table if not exists emails_test (id serial primary key, sender text not null, subject text, label integer not null)")
        return cursor
    else:
        # exit!
        print("Could not connect to PostgreSQL!")
        return connection


def fetch_emails(criteria, last_uid, next_uid, mail):
    msgs = []
    label = 1 if criteria == 'SEEN' else 0
    command = "{} UID {}:*".format(criteria, last_uid)
    result, data = mail.uid('search', None, command)
    if(result == "OK"):
        message_uids = data[0].split()
        for message_uid in message_uids:
            # search *always* returns at least the most recent message, so only fetch if we haven't already added the most recent message. if message_uid == 1, there's an exception, we want to add that message!
            if(int(message_uid) > last_uid or int(message_uid) == 1):
                result, data = mail.uid('fetch', message_uid, '(RFC822)')
                if(result == "OK"):
                    msg = email.message_from_bytes(data[0][1])
                    item = [msg['From'], msg['Subject'], label]
                    msgs.append(item)
                    if(int(message_uid) > next_uid):
                        next_uid = int(message_uid)
    return msgs, next_uid


def store_emails(emails, cursor, tablename):
    # convert array of lists to array of tuples for storing to db
    emails = [tuple(email) for email in emails]

    # only attempt to save if the array (of tuples) is not empty
    if emails:
        email_records = ", ".join(["%s"] * len(emails))
        try:
            cursor.execute(
                (f"insert into {tablename} (sender, subject, label) values {email_records}"), emails)
        except Exception as e:
            print(f'Exception: {e}\n')


def get_wordnet_pos(word):
    # find wordnet-pos from pos of word
    pos = nltk.pos_tag([word])[0][1]
    if pos.startswith('J'):
        return wordnet.ADJ
    elif pos.startswith('V'):
        return wordnet.VERB
    elif pos.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def process_subjects(subjects):
    new_subjects = []

    # define word lemmatizer
    lemmatizer = WordNetLemmatizer()

    # define regex to split subjects into tokens
    punc = re.compile("[" + string.punctuation + "\s" + "]")

    for subject in subjects:
        # 1) remove "Re:" from subject if it exists
        subject = subject.replace("Re:", "")

        # 2) split subject into tokens by punctuation/space
        parts = re.split(punc, subject)

        # 3) filter the tokens to create a new subject
        new_parts = []  # cleaned up parts, that will make up the new subject
        new_subject = ''  # cleaned up subject
        for part in parts:
            matches = re.match('[a-zA-Z]+', part)
            # keep a token if it consists of only alphabetic characters, and is not a single character
            if(matches and (matches.span()[1] - matches.span()[0] == len(part)) and len(part) > 1):
                # normalize
                part = part.lower()
                # lemmatize
                part = lemmatizer.lemmatize(part, get_wordnet_pos(part))
                # add the processed token(word)
                new_parts.append(part)
        new_subject = ' '.join(new_parts)
        new_subjects.append(new_subject)

    return new_subjects


def process_senders(senders):
    new_senders = []
    addr = re.compile(r"(\w)+(\S)+@(\S)+\.(\w)+")
    # only extract the email address from the sender field
    for sender in senders:
        matches = re.search(addr, sender)
        if(matches):
            new_sender = matches.group()
            new_senders.append(new_sender)
        else:
            new_senders.append(sender)
    return new_senders


def process_emails(emails):
    # preprocessing for sender & subject columns (nothing to do for the label)
    senders = [email[0] for email in emails]
    subjects = [email[1] for email in emails]
    new_senders = process_senders(senders)
    new_subjects = process_subjects(subjects)

    for i, email in enumerate(emails):
        email[0] = new_senders[i]
        email[1] = new_subjects[i]

    return emails


def split_test_train(emails, half_test_size):
    # test_indices: generate a random set of indices
    test_indices = random.sample(range(len(emails)), half_test_size)
    # train_indices: all indices except test indices
    train_indices = list(
        set(range(len(emails))).symmetric_difference(set(test_indices)))
    # create test & training subsets
    test_set = [emails[i] for i in test_indices]
    train_set = [emails[i] for i in train_indices]
    return test_set, train_set


def compute_test_train(read, unread):
    # calculate size of initial data
    total_size = len(read) + len(unread)
    # spare test set to be TEST_RATIO of the initial data
    test_size = total_size * TEST_RATIO
    # find the read & unread test size (the test set should contain 50% read, 50% unread emails!)
    half_test_size = math.floor(test_size / 2.)
    # split into read & unread - test & train
    read_test, read_train = split_test_train(read, half_test_size)
    unread_test, unread_train = split_test_train(unread, half_test_size)
    # combine test & train sets
    test_set = read_test + unread_test
    train_set = read_train + unread_train
    # shuffle the read/unread order in the sets
    random.shuffle(test_set)
    random.shuffle(train_set)
    # return test & train sets
    return test_set, train_set


def process_first_fetch(read, unread, cursor):
    # compute test & training sets
    testset, trainset = compute_test_train(read, unread)
    # store test & training sets to db
    store_emails(testset, cursor, "emails_test")
    store_emails(trainset, cursor, "emails")


def process_next_fetches(read, unread, cursor):
    # combine read & unread to a single list, shuffle the list
    trainset = read + unread
    random.shuffle(trainset)
    # store training set to db
    store_emails(trainset, cursor, "emails")
