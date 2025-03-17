# Dixit AI Computer Player
A simple AI agent as a player at the table for your Dixit game!

Install the requirements with:
```bash
$ pip3 install -r requirements.txt
```

The app is a simple Flask web application that can be run like this:
```bash
$ python3 app.py
```

The first time you run the app, Hugging Face's model will be downloaded. This is a large file and may take a while to download (~15GB, so about 10-15 minutes on a decent connection). 

You will be asked to upload an image of the card that you want to play. Currently this is a 2x3 grid of Dixit cards (that is the only format accepted, anything else will fail to break up the images correctly). Once you do this, and hit 'Upload and Start', the app will process the image and return a caption for one of the cards in AI agent's hand. 
