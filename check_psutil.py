import psutil

def find_procs():
    for conn in psutil.net_connections():
        if conn.laddr.port == 8000:
            print(f"FOUND: PID={conn.pid}, STATUS={conn.status}, LADDR={conn.laddr}")
            if conn.pid:
                proc = psutil.Process(conn.pid)
                print(f"CMDLINE: {proc.cmdline()}")

if __name__ == '__main__':
    find_procs()
