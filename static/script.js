document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const themeToggle = document.getElementById('theme-toggle');
    const body = document.body;
    
    const dbCountEl = document.getElementById('db-count');
    
    const fileInput = document.getElementById('file-input');
    const fileNameEl = document.getElementById('selected-file-name');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadForm = document.getElementById('upload-form');
    const uploadStatus = document.getElementById('upload-status');
    
    const topKInput = document.getElementById('top-k');
    const topKVal = document.getElementById('top-k-val');
    const poolInput = document.getElementById('pool');
    const poolVal = document.getElementById('pool-val');
    
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const clearChatBtn = document.getElementById('clear-chat');
    
    const loadingOverlay = document.getElementById('loading-overlay');

    // --- Theme Toggle ---
    themeToggle.addEventListener('click', () => {
        body.classList.toggle('light-mode');
        const icon = themeToggle.querySelector('i');
        if (body.classList.contains('light-mode')) {
            icon.classList.remove('fa-moon');
            icon.classList.add('fa-sun');
        } else {
            icon.classList.remove('fa-sun');
            icon.classList.add('fa-moon');
        }
    });

    // --- Settings Sliders ---
    topKInput.addEventListener('input', (e) => {
        topKVal.textContent = e.target.value;
    });
    
    poolInput.addEventListener('input', (e) => {
        poolVal.textContent = e.target.value;
    });

    // --- Auto-resize textarea ---
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // --- Fetch DB Status ---
    async function fetchStatus() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            dbCountEl.textContent = data.document_chunk_count;
        } catch (error) {
            dbCountEl.textContent = 'Hata';
        }
    }
    fetchStatus();

    // --- File Upload ---
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            fileNameEl.textContent = e.target.files[0].name;
            uploadBtn.disabled = false;
        } else {
            fileNameEl.textContent = '';
            uploadBtn.disabled = true;
        }
    });

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const file = fileInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Yükleniyor...';
        uploadStatus.textContent = '';
        uploadStatus.className = 'status-message';

        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if (res.ok) {
                uploadStatus.textContent = data.message;
                uploadStatus.classList.add('success');
                fileInput.value = '';
                fileNameEl.textContent = '';
                fetchStatus(); // Refresh count
            } else {
                throw new Error(data.detail || 'Yükleme başarısız.');
            }
        } catch (error) {
            uploadStatus.textContent = error.message;
            uploadStatus.classList.add('error');
        } finally {
            uploadBtn.disabled = true;
            uploadBtn.innerHTML = '<span>Yükle ve İndeksle</span><i class="fa-solid fa-arrow-right"></i>';
        }
    });

    // --- Chat Logic ---
    function appendMessage(role, content, sources = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerHTML = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-scale-balanced"></i>';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Handle markdown-like line breaks
        const formattedContent = content.replace(/\n/g, '<br>');
        contentDiv.innerHTML = `<p>${formattedContent}</p>`;
        
        if (sources && sources.length > 0) {
            const sourcesContainer = document.createElement('div');
            sourcesContainer.className = 'sources-container';
            
            const toggle = document.createElement('div');
            toggle.className = 'sources-toggle';
            toggle.innerHTML = `<i class="fa-solid fa-chevron-down"></i> ${sources.length} Kaynak Belge İnceltildi`;
            
            const sourcesContent = document.createElement('div');
            sourcesContent.className = 'sources-content hidden';
            
            sources.forEach((s, i) => {
                const card = document.createElement('div');
                card.className = 'source-card';
                card.innerHTML = `
                    <div class="source-header">
                        <span>Belge: ${s.doc_id}</span>
                        <span>Skor: ${s.rerank_score !== null ? s.rerank_score : s.fusion_score}</span>
                    </div>
                    <div class="source-text">${s.text}</div>
                `;
                sourcesContent.appendChild(card);
            });
            
            toggle.addEventListener('click', () => {
                sourcesContent.classList.toggle('hidden');
                const i = toggle.querySelector('i');
                if (sourcesContent.classList.contains('hidden')) {
                    i.classList.remove('fa-chevron-up');
                    i.classList.add('fa-chevron-down');
                } else {
                    i.classList.remove('fa-chevron-down');
                    i.classList.add('fa-chevron-up');
                }
            });
            
            sourcesContainer.appendChild(toggle);
            sourcesContainer.appendChild(sourcesContent);
            contentDiv.appendChild(sourcesContainer);
        }
        
        msgDiv.appendChild(avatar);
        msgDiv.appendChild(contentDiv);
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function askQuestion(query) {
        appendMessage('user', query);
        loadingOverlay.classList.remove('hidden');
        
        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    top_k: parseInt(topKInput.value),
                    pool: parseInt(poolInput.value)
                })
            });
            
            const data = await res.json();
            
            if (res.ok) {
                appendMessage('system', data.answer, data.sources);
            } else {
                throw new Error(data.detail || 'Bir hata oluştu.');
            }
        } catch (error) {
            appendMessage('system', `❌ Hata: ${error.message}`);
        } finally {
            loadingOverlay.classList.add('hidden');
        }
    }

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = chatInput.value.trim();
        if (!query) return;

        // Reset input
        chatInput.value = '';
        chatInput.style.height = 'auto';
        
        await askQuestion(query);
    });

    // --- Batch Questions ---
    const batchBtn = document.getElementById('batch-btn');
    const batchModal = document.getElementById('batch-modal');
    const closeBatchModal = document.getElementById('close-batch-modal');
    const startBatchBtn = document.getElementById('start-batch-btn');
    const questionsContainer = document.getElementById('batch-questions-container');
    const addQuestionBtn = document.getElementById('add-question-btn');

    function updateQuestionNumbers() {
        const rows = questionsContainer.querySelectorAll('.batch-q-row');
        rows.forEach((row, idx) => {
            row.querySelector('.q-number').textContent = (idx + 1) + '.';
            const delBtn = row.querySelector('.delete-q-btn');
            if (rows.length === 1) {
                delBtn.disabled = true;
            } else {
                delBtn.disabled = false;
            }
        });
    }

    addQuestionBtn.addEventListener('click', () => {
        const row = document.createElement('div');
        row.className = 'batch-q-row';
        row.innerHTML = `
            <span class="q-number"></span>
            <input type="text" class="batch-q-input" placeholder="Yeni sorunuz...">
            <button type="button" class="btn-icon delete-q-btn"><i class="fa-solid fa-trash"></i></button>
        `;
        
        row.querySelector('.delete-q-btn').addEventListener('click', () => {
            row.remove();
            updateQuestionNumbers();
        });
        
        questionsContainer.appendChild(row);
        updateQuestionNumbers();
        row.querySelector('.batch-q-input').focus();
    });

    questionsContainer.querySelector('.delete-q-btn').addEventListener('click', (e) => {
        e.currentTarget.closest('.batch-q-row').remove();
        updateQuestionNumbers();
    });

    batchBtn.addEventListener('click', () => {
        batchModal.classList.remove('hidden');
    });

    closeBatchModal.addEventListener('click', () => {
        batchModal.classList.add('hidden');
    });

    batchModal.addEventListener('click', (e) => {
        if (e.target === batchModal) {
            batchModal.classList.add('hidden');
        }
    });

    startBatchBtn.addEventListener('click', async () => {
        const inputs = questionsContainer.querySelectorAll('.batch-q-input');
        const questions = [];
        inputs.forEach(input => {
            const val = input.value.trim();
            if (val) questions.push(val);
        });
        
        if (questions.length === 0) return;
        
        // Reset modal
        questionsContainer.innerHTML = `
            <div class="batch-q-row">
                <span class="q-number">1.</span>
                <input type="text" class="batch-q-input" placeholder="Birinci sorunuz...">
                <button type="button" class="btn-icon delete-q-btn" disabled><i class="fa-solid fa-trash"></i></button>
            </div>
        `;
        questionsContainer.querySelector('.delete-q-btn').addEventListener('click', (e) => {
            e.currentTarget.closest('.batch-q-row').remove();
            updateQuestionNumbers();
        });
        
        batchModal.classList.add('hidden');
        
        // Process questions sequentially
        for (const query of questions) {
            await askQuestion(query);
        }
    });

    clearChatBtn.addEventListener('click', () => {
        chatMessages.innerHTML = `
            <div class="message system-message">
                <div class="avatar"><i class="fa-solid fa-robot"></i></div>
                <div class="message-content">
                    <p>Sohbet geçmişi temizlendi. Size nasıl yardımcı olabilirim?</p>
                </div>
            </div>
        `;
    });
});
