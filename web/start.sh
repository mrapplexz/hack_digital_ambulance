while true
do
gunicorn -c config.py app:server
done
