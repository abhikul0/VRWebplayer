import os
import re
import json
import http.server
import socketserver

PORT = 8000
VIDEO_EXTENSIONS = ('.mp4', '.m4v', '.webm', '.mov', '.mkv', '.avi')

class RangeHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_GET(self):
        if self.path == '/api/files':
            try:
                files = [f for f in os.listdir('.') if f.lower().endswith(VIDEO_EXTENSIONS) and os.path.isfile(f)]
                files.sort()
                response_data = json.dumps(files).encode('utf-8')
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Content-Length", str(len(response_data)))
                self.end_headers()
                self.wfile.write(response_data)
                return
            except Exception as e:
                self.send_error(500, str(e))
                return

        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().do_GET()
        
        try:
            f = open(path, 'rb')
        except OSError:
            self.send_error(404, "File not found")
            return

        file_size = os.fstat(f.fileno()).st_size
        range_header = self.headers.get('Range')

        if range_header:
            m = re.match(r"^bytes=(\d*)-(\d*)$", range_header)
            if m:
                g = m.groups()
                byte1 = int(g[0]) if g[0] else 0
                byte2 = int(g[1]) if g[1] else file_size - 1
                
                if byte1 >= file_size:
                    self.send_error(416, "Requested Range Not Satisfiable")
                    f.close()
                    return
                
                if byte2 >= file_size:
                    byte2 = file_size - 1
                    
                length = byte2 - byte1 + 1
                
                self.send_response(206)
                self.send_header("Content-type", self.guess_type(path))
                self.send_header("Content-Range", f"bytes {byte1}-{byte2}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                
                f.seek(byte1)
                remaining = length
                block_size = 65536
                try:
                    while remaining > 0:
                        chunk = f.read(min(remaining, block_size))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                    pass # Suppress normal browser disconnects during seeking/closing
                finally:
                    f.close()
                return

        self.send_response(200)
        self.send_header("Content-type", self.guess_type(path))
        self.send_header("Content-Length", str(file_size))
        self.end_headers()
        try:
            self.copyfile(f, self.wfile)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            f.close()

class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True

if __name__ == '__main__':
    server_address = ("", PORT)
    httpd = ThreadingHTTPServer(server_address, RangeHTTPRequestHandler)
    
    print(f"\n🚀 Robust VR Streaming Server Active!")
    print(f"📂 Serving directory: {os.path.abspath(os.getcwd())}")
    print(f"🔗 Open your browser at: http://localhost:{PORT}")
    print(f"📱 For Mobile/Quest: Use your local IP address (e.g., http://192.168.x.x:{PORT})")
    print("Press Ctrl+C to stop the server.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")