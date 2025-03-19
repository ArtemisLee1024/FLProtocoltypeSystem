// 初始化图表
const ctx = document.getElementById('metrics-chart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [1, 2, 3, 4, 5, 6, 7, 8, 9 ,10],
    datasets: [
      {
        label: 'Loss',
        data: [0.3427, 0.2015, 0.1499, 0.1221, 0.1052, 0.0952, 0.0856, 0.0799, 0.0749, 0.0707],
        borderColor: '#ff6384',
        yAxisID: 'y',
      },
      {
        label: 'Accuracy',
        data: [90.44, 94.30, 95.65, 96.38, 96.71, 97.06, 97.32, 97.44, 97.64, 97.74],
        borderColor: '#36a2eb',
        yAxisID: 'y1',
      }
    ]
  },
  options: {
    scales: {
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: { display: true, text: 'Loss' }
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        title: { display: true, text: 'Accuracy (%)' },
        grid: { drawOnChartArea: false }
      }
    }
  }
});

// 在app.js中添加动态显示逻辑
document.addEventListener('DOMContentLoaded', () => {
  // 训练模式切换
  document.getElementById('mode').addEventListener('change', function() {
    document.getElementById('hierarchical-group').classList.toggle('active', this.value === 'hierarchical');
  });

  // 差分隐私切换
  document.getElementById('dp').addEventListener('change', function() {
    document.getElementById('dp-params').classList.toggle('active', this.checked);
  });

  // 模型配置切换
  document.getElementById('model-type').addEventListener('change', function() {
    const isCustom = this.value === 'custom';
    document.getElementById('preset-model').classList.toggle('active', !isCustom);
    document.getElementById('custom-model').classList.toggle('active', isCustom);

    const nameInput = document.getElementById('model-name');
    if (this.value === 'preset') {
      const presetName = document.getElementById('model_size').options[document.getElementById('model_size').selectedIndex].text;
      nameInput.value = `${presetName}_${new Date().toLocaleDateString()}`;
    } else {
      if (nameInput.value.includes("预设")) {
        nameInput.value = "自定义模型_" + new Date().toLocaleDateString();
      }
    }
  });

  // CKKS加密需要自定义模型检查
  document.getElementById('encrypt').addEventListener('change', function() {
    if(this.value === 'ckks') {
      document.getElementById('model-type').value = 'custom';
      document.getElementById('model-type').dispatchEvent(new Event('change'));
    }
  });

  // 表单提交处理
  document.getElementById('config-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    // 获取模型名称（带默认值处理）
    let modelName = document.getElementById('model-name').value.trim();
    if (!modelName) {
      modelName = `未命名模型_${new Date().toISOString().slice(0, 16).replace(/[-T:]/g, '')}`;
    }
    // 更新监控面板显示
    document.getElementById('current-model').textContent = modelName;

    const config = {
      model_name: modelName,
      mode: document.getElementById('mode').value,
      clients: parseInt(document.getElementById('clients').value),
      rounds: parseInt(document.getElementById('rounds').value),
      local_epochs: parseInt(document.getElementById('local_epochs').value),
      dataset: document.getElementById('dataset').value,
      dp: document.getElementById('dp').checked,
      epsilon: parseFloat(document.getElementById('epsilon').value),
      delta: parseFloat(document.getElementById('delta').value),
      encrypt: document.getElementById('encrypt').value
    };
    // 处理分层参数
    if(config.mode === 'hierarchical') {
      config.hierarchical_groups = parseInt(document.getElementById('hierarchical_groups').value);
    }
    // 处理模型配置
    if(document.getElementById('model-type').value === 'custom') {
      const file = document.getElementById('model_config').files[0];
      if(file) {
        try {
          config.model_config = await handleFileUpload(file);
        } catch (error) {
          alert(error.message);
          return;
        }
      }
    } else {
      config.model_size = document.getElementById('model_size').value;
    }

    const eventSource = new EventSource(`/api/train?${Date.now()}`);
    eventSource.onmessage = (e) => {
      const data = JSON.parse(e.data);

      // 更新图表
      chart.data.labels.push(data.round);
      chart.data.datasets[0].data.push(data.loss);
      chart.data.datasets[1].data.push(data.accuracy * 100);
      chart.update();
    };
  });

  // 添加输入框实时验证
  document.getElementById('model-name').addEventListener('input', function() {
    this.value = this.value
      .replace(/[^a-zA-Z0-9_\u4e00-\u9fa5]/g, '') // 过滤特殊字符
      .slice(0, 30); // 限制最大长度
  });

  document.getElementById('validate-model-btn').addEventListener('click', () => {
    window.location.href = 'validate.html';  // 跳转到验证模型效果页面
  });
});

async function loadUserInfo() {
  try {
    const user = {
      avatar: "https://via.placeholder.com/40",
      name: "测试用户"
    };
//    const response = await fetch('/api/user');
//    const user = await response.json();
    document.querySelector('.username').textContent = user.name;
  } catch (error) {
    console.error('加载用户信息失败:', error);
  }
}

async function handleFileUpload(file) {
  const reader = new FileReader();
  return new Promise((resolve, reject) => {
    reader.onload = (e) => {
      try {
        const jsonData = JSON.parse(e.target.result);
        resolve(jsonData);  // 返回解析后的 JSON 对象
      } catch (error) {
        reject(new Error("文件解析失败，请上传有效的 JSON 文件"));
      }
    };
    reader.onerror = (error) => reject(error);
    reader.readAsText(file);
  });
}

// 加载模型列表
async function loadModels() {
  const response = await fetch('/api/history');
  const models = [{"name":"custom_mnist_20250301_150011.pth","size":"30.8KB","timestamp":1741762811.5154686},
  {"name":"custom_mnist_20250301_145222.csv","size":"0.4KB","timestamp":1741762811.511823},
  {"name":"custom_mnist_20250228_004548.pth","size":"30.8KB","timestamp":1741452348.8781862},
  {"name":"custom_mnist_20250228_004325.csv","size":"0.1KB","timestamp":1741452348.8747835},
  {"name":"custom_cifar10_20250228_004116.pth","size":"4108.5KB","timestamp":1741452076.0529206},
  {"name":"custom_cifar10_20250228_003346.csv","size":"0.2KB","timestamp":1741452076.0323887}];
//await response.json();

  const table = document.getElementById('model-table');
  table.innerHTML = models.map(model => `
    <div class="model-item">
      <div>${model.name}</div>
      <div>${model.size}</div>
      <div>${new Date(model.timestamp * 1000).toLocaleString()}</div>
      <a href="/api/models/${model.name}" download>⬇️ 下载</a>
      <a href="/api/models/delete/${model.name}"> ️✖️ 删除</a>
    </div>
  `).join('');
}

// 初始化加载
loadModels();
loadUserInfo();
setInterval(loadModels, 60);  // 每分钟刷新模型列表
