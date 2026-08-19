import socketio
import time
import json
import sys

# Standard python-socketio client
sio = socketio.Client(logger=True, engineio_logger=True)

@sio.event
def connect():
    print("Connected to BrainLotto API!")
    
    # Try sending some common events to see if it responds with Ghana data
    print("Testing common requests...")
    sio.emit('get_history', {'game': 'national', 'country': 'ghana'})
    sio.emit('get_ghana_data', {})
    sio.emit('get_charts', {'type': 'national'})

@sio.event
def disconnect():
    print("Disconnected from API.")

# Catch-all event listener to print literally anything the server sends us
@sio.on('*')
def catch_all(event, data):
    print(f"\n🚀 RECEIVED EVENT: {event}")
    
    # Save the data to a file in case it's huge
    filename = f"dump_{event}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print(f"Data saved to {filename}")
    
    # Print the first few lines of the data
    data_str = str(data)
    print(data_str[:500] + ("..." if len(data_str) > 500 else ""))

if __name__ == '__main__':
    try:
        url = "https://brainlotto-6b5ac0f18285.herokuapp.com"
        print(f"Connecting to {url}...")
        
        # Connect using polling/websocket
        sio.connect(url, transports=['websocket', 'polling'])
        
        print("Listening for 15 seconds...")
        time.sleep(15)
        
        sio.disconnect()
    except Exception as e:
        print(f"Error: {e}")
