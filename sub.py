import paho.mqtt.client as mqtt

BROKER = "localhost"   # atau IP broker Anda
PORT = 1883
TOPIC = "traffic/status"

def on_connect(client, userdata, flags, rc):
    print("Connected:", rc)
    client.subscribe(TOPIC)
    print("Subscribed ke", TOPIC)

def on_message(client, userdata, msg):
    print("Topic :", msg.topic)
    print("Payload :", msg.payload.decode())

client = mqtt.Client()

client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT)
client.loop_forever()