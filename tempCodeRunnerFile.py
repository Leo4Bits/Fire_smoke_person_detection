thread = threading.Thread(target=thread.serial_reader_thread, daemon=True)
thread.start()