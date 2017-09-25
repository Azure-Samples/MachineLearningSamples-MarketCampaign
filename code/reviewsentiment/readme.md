# Review Sentiment

Run BankReviewSentimentModeling.py in local environment.
```
$ az ml experiment submit -c local .\BankReviewSentimentModeling.py
```

Run BankReviewSentimentModeling.py in a local Docker container.
```
$ az ml experiment submit -c docker .\BankReviewSentimentModeling.py
```

Run BankReviewSentimentModeling.py in a Docker Container on a Remote Machine.
```
$ az ml computetarget attach --name myvm --address 52.187.129.184 --username ldsvmadmin --password <password> --type remotedocker
$ az ml experiment prepare -c myvm
$ az ml experiment submit -c myvm .\BankReviewSentimentModeling.py
```