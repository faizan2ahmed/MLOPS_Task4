install:
		pip install --no-cache-dir -r requirements.txt

docker-build:
		docker build -t task4 .
		
docker-run:
		docker run -p 5000:5000 task4
	