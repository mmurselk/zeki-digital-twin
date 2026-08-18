import roslibpy
import argparse
import time
 
message_count = 0
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test client for rosbridge websocket.")
    
    # Connection args
    parser.add_argument('--host', type=str, default='127.0.0.1', 
                        help='WebSocket host address.')
    parser.add_argument('--port', type=int, default=9090, 
                        help='WebSocket port (matches bridge.launch.py).')
    parser.add_argument('--ssl', action='store_true', 
                        help='Use wss:// for secure connections.')
    
    # Topic args
    parser.add_argument('--topic', type=str, required=True, 
                        help='Raw topic to subscribe to (e.g., /camera/image_raw).')
    parser.add_argument('--msg-type', type=str, required=True, 
                        help='ROS message type (e.g., sensor_msgs/Image).')
    
    # Test duration
    parser.add_argument('--duration', type=int, default=10, 
                        help='How long to keep the client alive (seconds).')
    
    return parser.parse_args()
    
 
 
def on_connect_handlers(client: roslibpy.Ros) -> None:
   # Triggered once the handshake completes
    client.on_ready(lambda: print("\n[SUCCESS] Connected to rosbridge server!"))
    
    # Catch silent connection failures
    client.on('error', lambda err: print(f"\n[ERROR] Connection error occurred: {err}"))
    client.on('close', lambda proto: print("\n[INFO] Connection to rosbridge closed."))

def subscribe_to_topic(client: roslibpy.Ros, topic_name: str, msg_type: str) -> roslibpy.Topic:
    topic = roslibpy.Topic(client, topic_name, msg_type)
    
    def callback(msg):
        global message_count
        message_count += 1
        current_time = time.strftime('%H:%M:%S')
        print(f"[{current_time}] Received message #{message_count} on {topic_name}")
        
    topic.subscribe(callback)
    print(f"[INFO] Subscribed to {topic_name} ({msg_type}). Waiting for messages...")
    return topic
 

 
def run_for_duration(client: roslibpy.Ros, seconds: int) -> None:
    print(f"[INFO] Running test for {seconds} seconds. Press Ctrl+C to abort.")
    start_time = time.time()
    
    try:
        # Loop until time is up, keeping the main thread alive 
        # while roslibpy handles callbacks in the background
        while client.is_connected and (time.time() - start_time) < seconds:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Test aborted by user (Ctrl+C).")


def print_summary(count: int, topic_name: str) -> None:
    print("\n" + "="*40)
    print("TEST SUMMARY")
    print("="*40)
    if count == 0:
        print(f"[WARNING] Received 0 messages on '{topic_name}'.")
        print("  -> Is the publisher running?")
        print("  -> Is the topic whitelisted in 'topics_glob' in bridge.launch.py?")
        print("  -> Are the host and port correct?")
    else:
        print(f"[SUCCESS] Received {count} messages on '{topic_name}'.")
        print("  -> The bridge is configured correctly and data is flowing.")
    print("="*40)


def main() -> None:
    args = parse_args()
    
    # Reset global counter in case main() is called multiple times in an interactive session
    global message_count
    message_count = 0
    
    # Instantiate client with ssl support
    client = roslibpy.Ros(host=args.host, port=args.port, is_secure=args.ssl)
    
    try:
        # Start connection in a background thread
        print(f"[INFO] Connecting to {'wss' if args.ssl else 'ws'}://{args.host}:{args.port}...")
        client = roslibpy.Ros(host=args.host, port=args.port, is_secure=args.ssl)
        on_connect_handlers(client)   # register handlers first
        client.run()                  # then connect
        
        # Give it a brief moment to establish connection before binding handlers
        timeout = 5
        start = time.time()
        while not client.is_connected and (time.time() - start) < timeout:
            time.sleep(0.1)
        if not client.is_connected:
            print("[ERROR] Failed to connect. Ensure bridge.launch.py is running.")
            sys.exit(1)
            
        on_connect_handlers(client)
        
        topic = subscribe_to_topic(client, args.topic, args.msg_type)
        
        run_for_duration(client, args.duration)
        
        print_summary(message_count, args.topic)
        
    finally:
        # Clean up connection gracefully
        if 'topic' in locals():
            topic.unsubscribe()
        if client.is_connected:
            client.terminate()


if __name__ == '__main__':
    main()