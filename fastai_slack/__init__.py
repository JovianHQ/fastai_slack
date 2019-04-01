from getpass import getpass
from fastai.callback import Callback
import traceback
import sys
import pandas as pd
import requests
import random
import torch
import string
import os
import json


def read_webhook_url(check_env=False, help_text=False):
    """Read the Slack incoming webhook URL from user input. See https://api.slack.com/incoming-webhooks

    Arguments:
        check_env (bool, optional): If true, attempts to read the URL from the 
        environment variable 'FASTAI_SLACK_WEBHOOK' before requesting user input.
    """
    ENV_KEY = 'FASTAI_SLACK_WEBHOOK'
    if ENV_KEY in os.environ:
        return os.environ[ENV_KEY]
    else:
        if help_text:
            print('You must provide a "Slack Incoming Webhook" URL for sending messages. ' +
                  'You can generate a webhook URL for your workspace by following this ' +
                  'guide: https://api.slack.com/incoming-webhooks')
        print('Enter webhook URL:')
        url = getpass()
        return url


def generate_tag():
    """Create a random string to uniquely identify jobs"""
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(6))


def sendNotification(url, msg):
    """Send a notification using a Slack webhook URL. See https://api.slack.com/incoming-webhooks

    Arguments:
        url (string): Slack incoming webhook URL for sending the message.
        msg (string): The message to be sent (can use markdown for formatting)
    """
    try:
        res = requests.post(url, data=json.dumps(
            {"text": msg, "mrkdwn": True}))
        if res.status_code != 200:
            print(f'Falied to send notification "{msg}" to "{url}"')
            print('Response', res.content)
            return False
    except Exception as e:
        print(f'Falied to send notification "{msg}" to "{url}"')
        print(e)
        return False
    return True


def format_metric(val):
    """Format a tensor/number as a float with 4 digits"""
    if isinstance(val, torch.Tensor):
        val = val.detach().data
    return str('{:.4f}'.format(val))


def format_metrics(names, metric_vals):
    """Print the metric names and values as a table"""
    vals = map(format_metric, metric_vals)
    msg = pd.DataFrame([vals], columns=names).to_string(index=False)
    return '```\n' + msg + '\n```'


class SlackCallback(Callback):
    """FastAI callback to send Slack notifications during/after training.

    Arguments:
        name (string): Project/job name (included in every slack notification)

        webhook_url (string, optional): Slack incoming webhook URL for sending
            messages to your workspace. You can generate a webhook URL for your 
            workspace by following this guide: https://api.slack.com/incoming-webhooks 
            If not provided, we try to read the environment variable 'FASTAI_SLACK_WEBHOOK'
            or prompt the user to provide the webhook URL as input.

        frequency (int, optional): No. of epochs after which a notifications 
            are sent. E.g. if frequency=10, notifications are sent every 10 epochs.
            Set frequency to 0 if you want a single notification at the end.
            By default, a notification is sent after every epoch.
    """

    def __init__(self, name, webhook_url=None, frequency=1):
        self.name, self.freq, self.tag = name, frequency, ''
        self.url = webhook_url or read_webhook_url(True, True)

    def _send(self, msg):
        """Send a notification with name and tag included"""
        if isinstance(msg, (list, tuple)):
            msg = '\n'.join(msg)
        sendNotification(self.url, f'[`{self.name} {self.tag}`] {msg}')

    def _send_metrics(self, ka, msg=None):
        """Format and send metrics for the current epoch"""
        msg = msg or []
        epoch, n_epochs = ka['epoch']+1, ka['n_epochs']
        epoch = min(epoch, n_epochs)
        # Log epoch no.
        msg .append(f'Epoch {epoch}/{n_epochs}')
        # Log table of metrics
        metrics_str = format_metrics(
            self.metrics, [ka['smooth_loss']] + ka['last_metrics'])
        msg.append(metrics_str)
        # Send message
        self._send(msg)

    def on_train_begin(self, **ka):
        "Called when training starts"
        # Generate a new tag
        self.tag = generate_tag()
        # Get the list of metric being tracked
        self.metrics = ['train_loss', 'valid_loss'] + ka['metrics_names']
        # Send message
        n_epochs = ka['n_epochs']
        self._send(f'*Started training for {n_epochs} epochs*')
        print('Slack notification tag:', f'[{self.name} {self.tag}]')

    def on_epoch_end(self, **ka):
        "Called at the end of an epoch."
        # Check whether to send metrics
        if self.freq > 0 and (ka['epoch']+1) % self.freq == 0:
            # Send notification
            self._send_metrics(ka)

    def on_train_end(self, **ka):
        "Called at the end of training"
        # Check for exception
        ex = ka['exception']
        if ex:
            # Format exception
            ex_str = f'`{ex}`'
            # Extract and format stacktrace
            if hasattr(sys, 'last_traceback'):
                tb = traceback.format_tb(getattr(sys, 'last_traceback'))
                tb_str = '```\n' + '\n'.join(tb) + '\n```'
            else:
                tb_str = 'Failed to detect stacktrace'
            # Send exception and stacktrace
            self._send(['*Training failed with exception:', ex_str, tb_str])
        else:
            # Log success
            msg = ['*Training complete*']
            # Check if final metrics should be reported
            n_epochs = ka['n_epochs']
            if self.freq == 0 or n_epochs % self.freq > 0:
                # Send notification
                self._send_metrics(ka, msg)
            else:
                self._send(msg)
