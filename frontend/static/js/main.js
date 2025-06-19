document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('resumeFiles');
    const fileList = document.getElementById('fileList');
    const screenButton = document.getElementById('screenButton');
    const jobDescription = document.getElementById('jobDescription');
    const results = document.getElementById('results');
    const resultsTable = document.getElementById('resultsTable');
    const loadingSpinner = document.getElementById('loadingSpinner');

    let uploadedFiles = new Map();

    // Drag and drop handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    // Handle file selection
    function handleFiles(files) {
        Array.from(files).forEach(file => {
            if (file.type === 'application/pdf' || 
                file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
                uploadedFiles.set(file.name, file);
                addFileToList(file);
            }
        });
        updateScreenButton();
    }

    // Add file to the list
    function addFileToList(file) {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <i class="fas fa-file-alt"></i>
            <span class="file-name">${file.name}</span>
            <span class="file-size">${formatFileSize(file.size)}</span>
            <button class="btn btn-sm btn-outline-danger" onclick="removeFile('${file.name}')">
                <i class="fas fa-times"></i>
            </button>
        `;
        fileList.appendChild(fileItem);
    }

    // Remove file from the list
    window.removeFile = function(fileName) {
        uploadedFiles.delete(fileName);
        const fileItems = fileList.getElementsByClassName('file-item');
        Array.from(fileItems).forEach(item => {
            if (item.querySelector('.file-name').textContent === fileName) {
                item.remove();
            }
        });
        updateScreenButton();
    };

    // Update screen button state
    function updateScreenButton() {
        screenButton.disabled = uploadedFiles.size === 0 || !jobDescription.value.trim();
    }

    // Format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Event listeners for job description
    jobDescription.addEventListener('input', updateScreenButton);

    // Screen resumes
    screenButton.addEventListener('click', async () => {
        if (uploadedFiles.size === 0 || !jobDescription.value.trim()) return;

        loadingSpinner.classList.remove('d-none');
        results.classList.add('d-none');
        resultsTable.innerHTML = '';

        try {
            const parsedResumes = [];
            
            // Parse each resume
            for (const [fileName, file] of uploadedFiles) {
                const formData = new FormData();
                formData.append('file', file);
                
                const response = await fetch('/parse-resume', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) throw new Error(`Failed to parse ${fileName}`);
                
                const result = await response.json();
                parsedResumes.push(result);
            }

            // Rank resumes
            const rankResponse = await fetch('/rank-resumes', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    job_description: { text: jobDescription.value },
                    resumes: parsedResumes
                })
            });

            if (!rankResponse.ok) throw new Error('Failed to rank resumes');

            const rankedResults = await rankResponse.json();
            displayResults(rankedResults);
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            loadingSpinner.classList.add('d-none');
        }
    });

    // Display results
    function displayResults(rankedResults) {
        results.classList.remove('d-none');
        
        rankedResults.forEach((result, index) => {
            const resume = result.resume;
            const score = result.similarity_score;
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${resume.name || 'N/A'}</td>
                <td>${resume.skills ? resume.skills.join(', ') : 'N/A'}</td>
                <td>
                    <span class="badge bg-primary score-badge">
                        ${(score * 100).toFixed(1)}%
                    </span>
                </td>
            `;
            resultsTable.appendChild(row);
        });
    }
}); 