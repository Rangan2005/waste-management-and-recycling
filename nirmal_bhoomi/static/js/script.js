const dropArea = document.querySelector('.drop-area');
const inputFile = document.getElementById('input-file');

dropArea.addEventListener('click', function () {
    inputFile.click();
});

inputFile.addEventListener('change', function () {
    const file = this.files[0];
    if (file.type.startsWith('image/')) {
        if (file.size < 8000000) {
            create_thumbnail(file);
            classifyImage(file); // Send image to backend after creating thumbnail
        } else {
            alert('Image size must be less than 8MB');
        }
    } else {
        alert('Must be image');
    }
});

dropArea.addEventListener('dragover', function (e) {
    e.preventDefault();
    this.style.borderStyle = 'solid';

    const h3 = this.querySelector('h3');
    h3.textContent = 'Release here to upload image';
});

dropArea.addEventListener('drop', function (e) {
    e.preventDefault();

    inputFile.files = e.dataTransfer.files;
    const file = e.dataTransfer.files[0];

    if (file.type.startsWith('image/')) {
        if (file.size < 8000000) {
            create_thumbnail(file);
            classifyImage(file); // Send image to backend after creating thumbnail
        } else {
            alert('Image size must be less than 8MB');
        }
    } else {
        alert('Must be image');
    }
});

const command = ['dragleave', 'dragend'];
command.forEach(item => {
    dropArea.addEventListener(item, function () {
        this.style.borderStyle = 'dashed';

        const h3 = this.querySelector('h3');
        h3.textContent = 'Drag and drop or click here to select image';
    });
});

function create_thumbnail(file) {
    const img = document.querySelectorAll('.thumbnail');
    const imgName = document.querySelectorAll('.img-name');
    img.forEach(item => item.remove());
    imgName.forEach(item => item.remove());

    const reader = new FileReader();
    reader.onload = () => {
        const url = reader.result;
        const img = document.createElement('img');
        img.src = url;
        img.className = 'thumbnail';
        const span = document.createElement('span');
        span.className = 'img-name';
        span.textContent = file.name;
        dropArea.appendChild(img);
        dropArea.appendChild(span);
        dropArea.style.borderColor = 'transparent';
    };
    reader.readAsDataURL(file);
}

// Function to get CSRF token
function getCSRFToken() {
    return document.querySelector('meta[name="csrf-token"]').getAttribute('content');
}

// Function to classify the image by sending it to the backend
function classifyImage(file) {
    const formData = new FormData();
    formData.append('image', file);

    fetch('/api/classify/', {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': getCSRFToken(), // Include CSRF token in headers
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = ''; // Clear previous results if any

        if (data.detections) {
            data.detections.forEach((detection, index) => {
                resultDiv.innerHTML += `
                    <p>Object ${index + 1}:</p>
                    <p>Classification: ${detection.classification}</p>
                `;
            });
        } else {
            resultDiv.innerHTML = '<p>No objects detected.</p>';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while processing the image.');
    });
}

