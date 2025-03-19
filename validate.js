async function loadModels() {
  const response = await fetch('/api/history');
  const models = [{"name":"custom_mnist_20250312_150011.pth","size":"30.8KB","timestamp":1741762811.5154686},
  {"name":"custom_mnist_20250309_004548.pth","size":"30.8KB","timestamp":1741452348.8781862},
  {"name":"custom_cifar10_20250309_004116.pth","size":"4108.5KB","timestamp":1741452076.0529206}];
//await response.json();

  const table = document.getElementById('model-list');
  table.innerHTML = models.map(model => `
    <div class="model-item">
      <div>${model.name}</div>
      <div>${model.size}</div>
      <div>${new Date(model.timestamp * 1000).toLocaleString()}</div>
    </div>
  `).join('');
}

document.addEventListener('DOMContentLoaded', () => {
  const modelList = document.getElementById('model-list');
  const selectedModelName = document.getElementById('selected-model-name');

  loadModels();
  // 模型选择逻辑
  modelList.addEventListener('click', (e) => {
    const modelItem = e.target.closest('.model-item');
    if (modelItem) {
      // 移除之前选中的样式
      document.querySelectorAll('.model-item.selected').forEach(item => {
        item.classList.remove('selected');
      });

      // 设置当前选中样式
      modelItem.classList.add('selected');
      const modelName = modelItem.querySelector('div').textContent;  // 获取模型名称
      selectedModelName.textContent = modelName;
    }
  });
});

document.getElementById('sample-upload').addEventListener('change', (e) => {
  const files = e.target.files;
  const preview = document.getElementById('sample-preview');
  preview.innerHTML = '';

  for (const file of files) {
    const reader = new FileReader();
    reader.onload = (event) => {
      const item = document.createElement('div');
      item.className = 'sample-item';

      if (file.type.startsWith('image/')) {
        item.innerHTML = `
          <img src="${event.target.result}" alt="${file.name}">
          <p>${file.name}</p>
        `;
      } else if (file.type === 'text/plain') {
        item.innerHTML = `
          <p>${file.name}</p>
          <p>${event.target.result}</p>  <!-- 显示文本内容 -->
        `;
      }

      preview.appendChild(item);
    };
    reader.readAsDataURL(file);
  }
});

document.getElementById('validate-btn').addEventListener('click', async () => {
  const files = document.getElementById('sample-upload').files;
  const selectedModel = document.querySelector('.model-item.selected')?.querySelector('div').textContent;

  if (!selectedModel) {
    alert('请选择一个模型');
    return;
  }

  if (files.length === 0) {
    alert('请上传至少一个样本');
    return;
  }

  const formData = new FormData();
  formData.append('model', selectedModel);
  for (const file of files) {
    formData.append('samples', file);
  }

  try {
    const response = await fetch('/api/validate', {
      method: 'POST',
      body: formData
    });
    const results = [
  {
    "sample": "img00002.jpg",
    "prediction": "car",
    "confidence": 0.93,
    "details": {
      "class1": "cat",
      "class2": "dog",
      "class3": "bird",
      "score1": 0.95,
      "score2": 0.03,
      "score3": 0.02
    }
  },
  {
    "sample": "img00006.jpg",
    "prediction": "positive",
    "confidence": 0.95,
    "details": {
      "class1": "positive",
      "class2": "neutral",
      "class3": "negative",
      "score1": 0.87,
      "score2": 0.10,
      "score3": 0.03
    }
  },
  {
    "sample": "img00008.jpg",
    "prediction": "horse",
    "confidence": 0.92,
    "details": {
      "class1": "positive",
      "class2": "neutral",
      "class3": "negative",
      "score1": 0.87,
      "score2": 0.10,
      "score3": 0.03
    }
  },
  {
    "sample": "img00010.jpg",
    "prediction": "cat",
    "confidence": 0.83,
    "details": {
      "class1": "positive",
      "class2": "neutral",
      "class3": "negative",
      "score1": 0.87,
      "score2": 0.10,
      "score3": 0.03
    }
  }
]//await response.json();

    const resultList = document.getElementById('result-list');
    resultList.innerHTML = results.map(result => `
      <div class="result-item">
        <div>样本: ${result.sample}</div>
        <div>结果: ${result.prediction}, 置信度: ${result.confidence}</div>
      </div>
    `).join('');
  } catch (error) {
    console.error('验证失败:', error);
    alert('验证失败，请重试');
  }
});