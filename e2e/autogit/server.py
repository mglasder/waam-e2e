# remote_server.py
from flask import Flask, request, jsonify
from e2e.autogit.autogit import git_add_commit_with

app = Flask(__name__)


@app.route("/execute", methods=["POST"])
def execute_code():
    # Extract data from the request
    data = request.json
    message = data.get("message", "")
    commit_hash = git_add_commit_with(message=message)
    return jsonify({"result": str(commit_hash)})


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=3000)
