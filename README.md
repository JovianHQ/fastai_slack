# Slack Notifications for FastAI

Get slack notifications for [FastAI](https://github.com/fastai/fastai) model training.

`fastai_slack` provides a simple callback to receive Slack notifcations while training FastAI models, with just one extra line of code. 

![](https://i.imgur.com/XkGHCFR.gif)


`fastai_slack` sends notifications to your chosen Slack workspace & channel for the following events:
* Start of training
* Losses and metrics at the end of every epoch (or every few epochs)
* End of training
* Exceptions that occur during training (with stack trace)

## Installation and Usage

See this Jupyter notebook for a detailed setup and usage example: https://jvn.io/aakashns/2e505d1a5e7b49e18e7849da3280b9da 

1. Install the `fastai_slack` library using pip:

```
pip install fastai_slack
```

2. Generate a [Slack incoming webhook URL](https://api.slack.com/incoming-webhooks), which will allow you to send notifications to a Slack channel of your choice. More details here: https://api.slack.com/incoming-webhooks

The webhook URL should be kept secret, so `fastai_slack` provides a helpful `read_webhook_url` method to safely input webhook URL within Jupyter.

```python
from fastai_slack import read_webhook_url
webhook_url = read_webhook_url()
```

3. While calling the `fit` method to train the model, passing in a `SlackCallback` configured with:

`name`: project/job name which will be included in every notification
`webhook_url`: The Slack incoming webhook URL (read from user input earlier)
`frequency`: How often to send a notification (defaults to 1 i.e. every epoch)

Here's a complete example:

```
from fastai.vision import *
from fastai_slack import SlackCallback

# Create a learner
data = ImageDataBunch.from_folder(untar_data(URLs.MNIST),train='training', valid='testing')
learn = cnn_learner(data, models.resnet18, metrics=accuracy)

# Create a callback
slack_cb = SlackCallback('mnist', webhook_url, frequency=2)

# fit with callback
learn.fit(8, callbacks=[slack_cb])
```

Here's what the notifications look like:

![](https://i.imgur.com/ANQiZEp.png)

## Configuration

Instead of passing the webhook URL manually each time, you can also set a system wide environment variable with the name `FASTAI_SLACK_WEBHOOK` contaning the URL, and it will be read automatically by `SlackCallback`. For instance, on Mac/Linux, you might need to add the following like to your `~/.bashrc`:

```
FASTAI_SLACK_WEBHOOK=https://hooks.slack.com/services/T00000/B0000/XXXXXXXXXXX
# Replace the URL above with your Slack incoming webhook URL
```

## Contributing
fastai_slack is open source, and you can view the source code here: https://github.com/swiftace-ai/fastai_slack

Please use Github to report bugs, request/propose feature. Pull requests with bug fixes and PRs are most welcome!
