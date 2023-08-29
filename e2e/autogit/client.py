# client.py
import requests


def main():
    # The code you want to execute on the remote machine
    message = "test_run_id"
    response = requests.post("http://10.6.6.55:3000/execute", json={"message": message})
    print(response.json()["result"])


if __name__ == "__main__":
    main()
