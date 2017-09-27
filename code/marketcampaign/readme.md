# Market Campaign

Run BankMarketCampaignModeling.py in local environment.
```
$ az ml experiment submit -c local .\BankMarketCampaignModeling.py
```

Run BankMarketCampaignModelingDocker.py in a local Docker container.
```
$ az ml experiment submit -c docker .\BankMarketCampaignModelingDocker.py
```

Run BankMarketCampaignModelingDocker.py in a Docker Container on a Remote Machine.
```
$ az ml computetarget attach --name myvm --address 52.187.129.184 --username ldsvmadmin --password <password> --type remotedocker
$ az ml experiment prepare -c myvm
$ az ml experiment submit -c myvm .\BankMarketCampaignModelingDocker.py
```