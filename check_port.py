import subprocess
import sys

def run():
    try:
        res = subprocess.run(['ss', '-lptn'], capture_output=True, text=True)
        with open('port_8000_check.txt', 'w') as f:
            for line in res.stdout.splitlines():
                if '8000' in line or 'State' in line:
                    f.write(line + '\n')
    except Exception as e:
        with open('port_8000_check.txt', 'w') as f:
            f.write(str(e))

if __name__ == '__main__':
    run()
