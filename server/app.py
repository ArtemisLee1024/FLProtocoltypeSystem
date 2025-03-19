import datetime
import json
import os
import shutil
import subprocess
import uuid

import jwt
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
MODEL_DIR = 'saved_models'
LOG_DIR = 'saved_logs'
os.makedirs(MODEL_DIR, exist_ok=True)

# 配置密钥（生产环境应使用更安全的存储方式）
app.config['SECRET_KEY'] = 'your-secret-key-here'


# 模拟用户数据库
class UserDB:
    def __init__(self):
        self.users = {
            'Artemis': {
                'password': 'admin123',
                'client_id': 0
            }
        }
        self.client_map = {}  # 记录client_id占用情况

    def add_user(self, username, password, client_id):
        if username in self.users:
            return False
        if client_id in self.client_map:
            return False
        self.users[username] = {
            'password': password,
            'client_id': client_id
        }
        self.client_map[client_id] = username
        return True


user_db = UserDB()


@app.route('/login', methods=['POST'])
def login():
    auth_data = request.get_json()

    username = auth_data.get('username')
    password = auth_data.get('password')
    client_id = auth_data.get('client_id')

    user = user_db.users.get(username)

    # 验证用户是否存在
    if not user:
        return jsonify({'error': '用户不存在'}), 401

    # 验证密码
    if user['password'] != password:
        return jsonify({'error': '密码错误'}), 401

    # 验证客户端ID
    if user.get('client_id') != client_id:
        return jsonify({'error': '客户端ID不匹配'}), 403

    # 生成JWT Token
    token = jwt.encode({
        'username': username,
        'client_id': client_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    }, app.config['SECRET_KEY'], algorithm='HS256')

    return jsonify({
        'message': '登录成功',
        'token': token,
        'user': {
            'username': username,
            'client_id': client_id
        }
    })


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    username = data.get('username')
    password = data.get('password')
    client_id = data.get('client_id')

    # 输入验证
    if not all([username, password, client_id]):
        return jsonify({'error': '所有字段必须填写'}), 400

    # 注册用户
    if not user_db.add_user(username, password, client_id):
        return jsonify({'error': '用户名或客户端ID已被占用'}), 409

    return jsonify({'message': '用户注册成功'})


@app.route('/model/train', methods=['POST'])
def start_training():
    try:
        # 解析并验证请求参数
        config = request.json
        validate_config(config)  # 自定义验证函数

        # 生成临时配置文件路径
        config_id = str(uuid.uuid4())
        temp_dir = "temp_configs"
        os.makedirs(temp_dir, exist_ok=True)
        model_config_path = os.path.join(temp_dir, f"{config_id}.json")

        # 处理 model_config
        if 'model_config' in config:
            # 如果 model_config 是文件内容（JSON 字符串）
            if isinstance(config['model_config'], dict):
                with open(model_config_path, 'w') as f:
                    json.dump(config['model_config'], f)
            # 如果 model_config 是文件路径
            elif isinstance(config['model_config'], str):
                if not os.path.exists(config['model_config']):
                    raise ValueError("指定的模型配置文件不存在")
                model_config_path = config['model_config']
            else:
                raise ValueError("model_config 必须是 JSON 对象或文件路径")

        # 构建命令行参数
        cmd = ["python", "../server/train/core.py"]
        args_mapping = {
            'mode': ('--mode', str),
            'hierarchical_groups': ('--hierarchical_groups', int),
            'clients': ('--clients', int),
            'rounds': ('--rounds', int),
            'local_epochs': ('--local_epochs', int),
            'dataset': ('--dataset', str),
            'epsilon': ('--epsilon', float),
            'delta': ('--delta', float),
            'encrypt': ('--encrypt', str),
            'model_config': ('--model_config', str),
            'model_size': ('--model_size', str)
        }

        for key in config:
            if key in args_mapping:
                arg_name, type_cast = args_mapping[key]
                if key == 'model_config':
                    cmd.extend([arg_name, model_config_path])
                else:
                    cmd.extend([arg_name, str(type_cast(config[key]))])

        # 处理布尔型参数
        if config.get('dp', False):
            cmd.append('--dp')

        # 启动训练进程
        def generate():
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            while True:
                output = proc.stdout.readline()
                if output == '' and proc.poll() is not None:
                    break
                if output:
                    yield f"{output}\n\n"

            # 清理临时文件
            if os.path.exists(model_config_path):
                os.remove(model_config_path)

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


@app.route('/history')
def list_models():
    result = []
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith('.pth'):
            stat = os.stat(os.path.join(MODEL_DIR, fname))
            result.append({
                'name': fname,
                'size': f"{stat.st_size / 1024:.1f}KB",
                'timestamp': stat.st_mtime
            })
    for fname in os.listdir(LOG_DIR):
        if fname.endswith('.csv'):
            stat = os.stat(os.path.join(LOG_DIR, fname))
            result.append({
                'name': fname,
                'size': f"{stat.st_size / 1024:.1f}KB",
                'timestamp': stat.st_mtime
            })
    return jsonify(sorted(result, key=lambda x: -x['timestamp']))


@app.route('/download/model/<model_name>')
def download_model(model_name):
    return send_from_directory(MODEL_DIR, model_name, as_attachment=True)


@app.route('/download/record/<record_name>')
def download_model(record_name):
    return send_from_directory(LOG_DIR, record_name, as_attachment=True)


# 在app.py中添加以下路由
@app.route('/model/validate', methods=['POST'])
def validate_model():
    try:
        # 获取上传的模型和样本
        model_name = request.form.get('model')
        if not model_name:
            return jsonify({"error": "未选择模型"}), 400

        # 验证模型文件是否存在
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            return jsonify({"error": "模型文件不存在"}), 404

        # 获取上传的样本文件
        samples = request.files.getlist('samples')
        if not samples:
            return jsonify({"error": "未上传样本"}), 400

        # 创建临时目录保存样本
        temp_dir = os.path.join("temp_uploads", str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)

        saved_files = []
        for file in samples:
            if file.filename == '':
                continue
            file_path = os.path.join(temp_dir, file.filename)
            file.save(file_path)
            saved_files.append(file_path)

        # 调用验证函数（这里需要实现实际验证逻辑）
        results = run_validation(model_path, saved_files)
        # 清理临时文件
        shutil.rmtree(temp_dir)
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run_validation(model_path, sample_paths):
    """模拟验证过程，实际应替换为真实模型推理"""
    results = []
    return results


def validate_config(config):
    """配置验证逻辑"""
    required_fields = ['mode', 'clients', 'rounds', 'dataset']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    if config['mode'] == 'hierarchical' and 'hierarchical_groups' not in config:
        raise ValueError("hierarchical_groups is required for hierarchical mode")

    if config.get('encrypt') == 'ckks' and config.get('model_config') is None:
        raise ValueError("CKKS encryption requires custom model configuration")

    if config['dataset'] not in ['mnist', 'cifar10', 'fashion']:
        raise ValueError("Invalid dataset selection")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
