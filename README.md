## Smartreply
##### Smart Reply A.I using Deep Learning and NLP techniques.
-----------------------------------

#### Setting up the environment 

- Use `virtualenv` (install using `apt-get install virtualenv`)
- Install python3.8 (`apt-get install python3.8`)
- Create virtual environment using `virtualenv venv --python=python3.8`
- Activate virtualenv using `source venv/bin/activate`
- use `python api.py` for testing

#### Training the A.I Model

- Update the `intents.json` file with new training data
- Run `python training.py`
- Model file will be generated and will be saved as `smart-reply-data.pkl`
- Thats it!

#### Using the Flask-API

- Run `python api.py`
- Use cURL or Postman to test the api running in port 80



### cURL request

#### Predict smart response
        curl --location --request POST 'http://127.0.0.1/api/v1/smartreply' \
        --header 'Content-Type: application/json' \
        --data-raw '{
            "sentence":"Very bad restaurant",
            "username":"Jason",
            "email":"support@mail.com",
            "phone":"9999999999",
            "website":"https://example.com"

        }'

#### Add more training data
        curl --location --request POST 'http://127.0.0.1/api/v1/update' \
        --header 'Content-Type: application/json' \
        --data-raw '{
            "pattern": "st22323",
            "tag": "spam"
        }'

#### Add more response
        curl --location --request POST 'http://127.0.0.1/api/v1/addresponse' \
        --header 'Content-Type: application/json' \
        --data-raw '{
            "response": "Hello {user}, please specify your concern",
            "tag": "spam"
        }'
