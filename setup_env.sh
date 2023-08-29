# create env uisng venv
python3 -m venv env 

# activate env
source ./env/bin/activate

python -m pip install ipykernel
python -m ipykernel install --user --name=env
echo Done!

python -m pip install --upgrade pip

# install requirements
python -m pip install -r requirements.txt
