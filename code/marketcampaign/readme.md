# Market Campaign

Run BankMarketCampaignModeling.py in local environment.
```
$ az ml experiment submit -c local .\code\marketcampaign\BankMarketCampaignModeling.py
```

Run BankMarketCampaignModelingDocker.py in a local Docker container.
```
$ az ml experiment submit -c docker .\code\marketcampaign\BankMarketCampaignModelingDocker.py
```

Run BankMarketCampaignModelingDocker.py in a Docker Container on a Remote Machine.
```
$ az ml computetarget attach remotedocker --name "myvm" --address "<id address>" --username "<username>" --password "<password>"
$ az ml experiment prepare -c myvm
$ az ml experiment submit -c myvm .code\marketcampaign\BankMarketCampaignModelingDocker.py
```