        const form = document.getElementById('chat-form');
        const input = document.getElementById('user-input');
        const chatWindow = document.getElementById('chat-window');
        const messages = document.getElementById('messages');
        const sendBtn = document.getElementById('send-btn');
        const resetBtn = document.getElementById('reset-btn');
        const helpBtn = document.getElementById('help-btn');
        const sidebar = document.querySelector('.sidebar');
        const sidebarClose = document.getElementById('sidebar-close');
        const sidebarBackdrop = document.getElementById('sidebar-backdrop');
        const exampleChips = document.querySelectorAll('.example-chip');
        const heatmapModal = document.getElementById('heatmap-modal');
        const heatmapBackdrop = document.getElementById('heatmap-backdrop');
        const heatmapClose = document.getElementById('heatmap-close');
        const heatmapContent = document.getElementById('heatmap-content');

        function setAppHeight() {
            const viewportHeight = window.visualViewport
                ? window.visualViewport.height
                : window.innerHeight;

            document.documentElement.style.setProperty('--app-height', `${viewportHeight}px`);
        }

        function openSidebar() {
            if (window.innerWidth > 960) return;

            sidebar.classList.add('is-open');
            sidebarBackdrop.classList.add('is-open');
            document.body.classList.add('sidebar-open');
            helpBtn?.setAttribute('aria-expanded', 'true');
        }

        function closeSidebar() {
            sidebar.classList.remove('is-open');
            sidebarBackdrop.classList.remove('is-open');
            document.body.classList.remove('sidebar-open');
            helpBtn?.setAttribute('aria-expanded', 'false');
        }

        setAppHeight();
        window.addEventListener('resize', setAppHeight);

        if (window.visualViewport) {
            window.visualViewport.addEventListener('resize', setAppHeight);
            window.visualViewport.addEventListener('scroll', setAppHeight);
        }

        window.addEventListener('resize', () => {
            if (window.innerWidth > 960) {
                closeSidebar();
            }
        });

        const userId = localStorage.getItem('chat_user_id') || (() => {
            const id = `user_${crypto.randomUUID()}`;
            localStorage.setItem('chat_user_id', id);
            return id;
        })();

        let currentConversationId = localStorage.getItem('current_conversation_id') || '';

        async function ensureConversationReady() {
            if (currentConversationId) return currentConversationId;

            const response = await fetch('/conversations', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: userId,
                    title: '새 채팅'
                })
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data?.detail || '대화방 생성 오류');
            }

            currentConversationId = data.conversation.conversation_id;
            localStorage.setItem('current_conversation_id', currentConversationId);
            return currentConversationId;
        }

        function autoResizeTextarea() {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 180) + 'px';
        }

        function scrollToBottom(forceSmooth = false) {
            if (forceSmooth) {
                chatWindow.scrollTo({
                    top: chatWindow.scrollHeight,
                    behavior: 'smooth'
                });
                return;
            }

            chatWindow.scrollTop = chatWindow.scrollHeight;
        }

        input.addEventListener('input', autoResizeTextarea);

        input.addEventListener('focus', () => {
            setAppHeight();
            setTimeout(() => {
                scrollToBottom();
            }, 250);
        });

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                form.requestSubmit();
            }
        });

        helpBtn?.addEventListener('click', openSidebar);
        sidebarClose?.addEventListener('click', closeSidebar);
        sidebarBackdrop?.addEventListener('click', closeSidebar);

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeSidebar();
            }
        });

        exampleChips.forEach((chip) => {
            chip.addEventListener('click', () => {
                const chipText = chip.textContent.trim();
                let template = "";

                // 버튼 이름별 맞춤형 템플릿 설정
                switch (chipText) {
                    case '자격증 지원':
                        template = `자격증 지원 프로그램이 궁금해.\n지역:\n나이:\n취업 여부:`;
                        break;
                    case '월세 지원':
                        template = `월세 지원 정책에 대해 알려줘.\n지역:\n나이:\n소득수준:`;
                        break;
                    case '취업 지원':
                        template = `취업 지원 프로그램에 대해 알려줘.\n관심 직무:\n지역:\n현재 상태(학생/준비생):\n희망 기업 규모:`;
                        break;
                    case '창업 지원':
                        template = `청년 창업 지원금에 대해 궁금해.\n나이:\n희망 창업 분야:\n사업자 등록 여부:`;
                        break;
                    case '자산 형성':
                        template = `청년을 위한 자산 형성 지원 정책이 궁금해.\n현재 직업(재직/준비생):\n월 평균 소득:\n거주 지역:\n중위소득 비율:`;
                        break;
                    default:
                        template = `${chipText} 관련 정책 알려줘.`;
                }

                // 입력창에 템플릿 삽입 및 UI 업데이트
                input.value = template;
                autoResizeTextarea();
                input.focus();

                // 모바일 브라우저일 경우 사이드바 닫기
                if (window.innerWidth <= 960) {
                    closeSidebar();
                }

                // 입력 내용이 잘 보이도록 하단으로 스크롤
                setTimeout(() => {
                    scrollToBottom(true);
                }, 120);
            });
        });

        function getTimeLabel() {
            const now = new Date();
            return now.toLocaleTimeString('ko-KR', {
                hour: '2-digit',
                minute: '2-digit'
            });
        }

        function createTypingIndicator() {
            const wrap = document.createElement('div');
            wrap.className = 'typing';
            wrap.innerHTML = '<span></span><span></span><span></span>';
            return wrap;
        }

        function appendMessage(sender, text, isLoading = false) {
            const id = 'msg-' + Date.now() + '-' + Math.floor(Math.random() * 10000);
            const isUser = sender === 'user';

            const row = document.createElement('div');
            row.className = `message-row ${isUser ? 'message-row--user' : 'message-row--bot'} fade-in`;
            row.id = id;

            const avatar = document.createElement('div');
            avatar.className = `avatar ${isUser ? 'avatar--user' : 'avatar--bot'}`;
            avatar.textContent = isUser ? 'ME' : 'AI';

            const wrap = document.createElement('div');
            wrap.className = 'bubble-wrap';

            const meta = document.createElement('div');
            meta.className = 'message-meta';
            meta.textContent = isUser ? `나 · ${getTimeLabel()}` : `정책 상담 AI · ${getTimeLabel()}`;

            const bubble = document.createElement('div');
            bubble.className = `bubble ${isUser ? 'bubble--user' : 'bubble--bot'}${isLoading ? ' loading' : ''}`;

            if (isLoading) {
                bubble.appendChild(createTypingIndicator());
            } else {
                bubble.textContent = text;
            }

            wrap.appendChild(meta);
            wrap.appendChild(bubble);

            if (isUser) {
                row.appendChild(wrap);
                row.appendChild(avatar);
            } else {
                row.appendChild(avatar);
                row.appendChild(wrap);
            }

            messages.appendChild(row);
            scrollToBottom();
            return id;
        }

        function renderMarkdown(text) {
            const rawHtml = marked.parse(text || '', {
                breaks: true
            });
            return DOMPurify.sanitize(rawHtml);
        }

        function escapeHtml(text) {
            return text
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
        }

        function getWordHighlightColor(score, rank) {
            if (rank === 0) return 'rgba(255,230,120,0.6)';
            if (rank <= 2) return 'rgba(255,255,255,0.30)';
            if (score >= 0.35) return 'rgba(255,255,255,0.20)';
            return 'transparent';
        }

        function buildUserHighlightedHtml(originalMessage, captumResult) {
            if (!captumResult || !Array.isArray(captumResult.word_scores)) {
                return `<div>${escapeHtml(originalMessage)}</div>`;
            }

            const words = [...captumResult.word_scores]
                .filter(w => w && typeof w.start === 'number' && typeof w.end === 'number')
                .sort((a, b) => a.start - b.start);

            if (words.length === 0) {
                return `<div>${escapeHtml(originalMessage)}</div>`;
            }

            const ranked = [...words]
                .map((w, idx) => ({ ...w, originalIndex: idx }))
                .sort((a, b) => b.score - a.score);

            const rankMap = new Map();
            ranked.forEach((w, i) => {
                rankMap.set(`${w.start}-${w.end}`, i);
            });

            let html = `<div class="user-highlight">`;
            let cursor = 0;

            for (const w of words) {
                const plainText = originalMessage.slice(cursor, w.start);
                if (plainText) {
                    html += escapeHtml(plainText);
                }

                const key = `${w.start}-${w.end}`;
                const rank = rankMap.get(key) ?? 999;
                const bg = getWordHighlightColor(w.score, rank);
                const wordText = originalMessage.slice(w.start, w.end);

                if (bg === 'transparent') {
                    html += escapeHtml(wordText);
                } else {
                    html += `<span class="user-highlight-word" style="background:${bg}" title="score=${w.score.toFixed(4)}">${escapeHtml(wordText)}</span>`;
                }

                cursor = w.end;
            }

            if (cursor < originalMessage.length) {
                html += escapeHtml(originalMessage.slice(cursor));
            }

            html += `</div>`;
            html += `<div class="user-highlight-note">Captum 기준으로 입력 질문에서 상대적으로 중요한 단어를 색으로 표시했습니다.</div>`;

            return html;
        }

        function updateUserMessageWithCaptum(id, originalMessage, captumResult) {
            const row = document.getElementById(id);
            if (!row) return;

            const bubble = row.querySelector('.bubble');
            if (!bubble) return;

            bubble.innerHTML = buildUserHighlightedHtml(originalMessage, captumResult);
            scrollToBottom();
        }

        function openHeatmapModal(originalMessage, captumResult) {
            heatmapContent.innerHTML = buildHeatmapModalHtml(originalMessage, captumResult);
            heatmapModal.hidden = false;
            document.body.classList.add('sidebar-open');
        }

        function closeHeatmapModal() {
            heatmapModal.hidden = true;
            heatmapContent.innerHTML = '';
            document.body.classList.remove('sidebar-open');
        }

        heatmapClose?.addEventListener('click', closeHeatmapModal);
        heatmapBackdrop?.addEventListener('click', closeHeatmapModal);

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !heatmapModal.hidden) {
                closeHeatmapModal();
            }
        });

        function getHeatColor(score) {
            const alpha = Math.max(0.12, Math.min(score, 1) * 0.75);
            return `rgba(37, 99, 235, ${alpha})`;
        }

        function buildHeatStripHtml(wordScores) {
            return `
                <div class="heat-strip">
                    ${wordScores.map(item => `
                        <span
                            class="heat-strip__token"
                            style="background:${getHeatColor(item.score)}"
                            title="${item.word} / score=${item.score.toFixed(4)}"
                        >
                            ${escapeHtml(item.word)}
                        </span>
                    `).join('')}
                </div>
            `;
        }

        function buildBarChartHtml(wordScores) {
            const sorted = [...wordScores].sort((a, b) => b.score - a.score);

            return `
                <div class="bar-chart">
                    ${sorted.map(item => `
                        <div class="bar-row">
                            <div class="bar-row__label">${escapeHtml(item.word)}</div>
                            <div class="bar-row__track">
                                <div class="bar-row__fill" style="width:${Math.max(item.score * 100, 4)}%"></div>
                            </div>
                            <div class="bar-row__value">${(item.score * 100).toFixed(1)}%</div>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        function buildHeatmapModalHtml(originalMessage, captumResult) {
            if (!captumResult || !Array.isArray(captumResult.word_scores) || captumResult.word_scores.length === 0) {
                return `
                    <div class="heatmap-panel">
                        <div class="heatmap-question">${escapeHtml(originalMessage)}</div>
                        <div>분석 결과가 없습니다.</div>
                    </div>
                `;
            }

            const wordScores = captumResult.word_scores.filter(item => item && item.word);

            return `
                <div class="heatmap-panel">
                    <div class="heatmap-question">${escapeHtml(originalMessage)}</div>
                    ${buildHeatStripHtml(wordScores)}
                    ${buildBarChartHtml(wordScores)}
                </div>
            `;
        }

        function buildHeatmapButton(originalMessage, captumResult) {
            if (!captumResult || !Array.isArray(captumResult.word_scores) || captumResult.word_scores.length === 0) {
                return null;
            }

            const wrap = document.createElement('div');
            wrap.className = 'xai-action-row';

            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'heatmap-open-btn';
            button.innerHTML = '<i class="fa-solid fa-fire"></i><span>입력값 중요도 그래프 (히트맵) 보기</span>';

            button.addEventListener('click', () => {
                openHeatmapModal(originalMessage, captumResult);
            });

            wrap.appendChild(button);
            return wrap;
        }


        function updateMessage(id, text, originalMessage = '', captumData = null) {
            const row = document.getElementById(id);
            if (!row) return;

            const bubble = row.querySelector('.bubble');
            if (!bubble) return;

            bubble.classList.remove('loading');
            bubble.innerHTML = '';

            const textNode = document.createElement('div');
            textNode.className = 'markdown-body';
            textNode.innerHTML = renderMarkdown(text);
            bubble.appendChild(textNode);

            const stack = document.createElement('div');
            stack.className = 'analysis-stack';

            const heatmapButton = buildHeatmapButton(originalMessage, captumData);
            if (heatmapButton) {
                stack.appendChild(heatmapButton);
            }

            if (stack.children.length > 0) {
                bubble.appendChild(stack);
            }

            scrollToBottom();
        }

        async function callChatAPI(message) {
            const conversationId = await ensureConversationReady();

            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: userId,
                    conversation_id: conversationId,
                    message
                })
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data?.detail || '채팅 응답 오류');
            }
            return data;
        }

        async function callCaptumAPI(message) {
            const conversationId = await ensureConversationReady();

            const response = await fetch('/captum', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: userId,
                    conversation_id: conversationId,
                    message
                })
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data?.detail || 'Captum 응답 오류');
            }
            return data;
        }

        async function sendMessage(message, userMessageId) {
            const loadingId = appendMessage('bot', '', true);

            try {
                const [chatResult, captumResult] = await Promise.allSettled([
                    callChatAPI(message),
                    callCaptumAPI(message)
                ]);

                if (chatResult.status !== 'fulfilled') {
                    throw chatResult.reason;
                }

                const chatData = chatResult.value;
                const captumData = captumResult.status === 'fulfilled' ? captumResult.value : null;

                if (captumData) {
                    console.log('=== Captum 단어별 기여도 ===');
                    (captumData.word_scores || []).forEach((item, idx) => {
                        console.log(
                            `${idx + 1}. ${item.word} | score=${item.score} | span=(${item.start}, ${item.end})`
                        );
                    });

                    updateUserMessageWithCaptum(userMessageId, message, captumData);
                }

                updateMessage(
                    loadingId,
                    chatData.answer || '응답을 생성하지 못했습니다.',
                    message,
                    captumData
                );

                if (captumResult.status === 'rejected') {
                    console.warn('Captum 분석 실패:', captumResult.reason);
                }
            } catch (error) {
                updateMessage(
                    loadingId,
                    '서버 응답 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.'
                );
                console.error(error);
            } finally {
                sendBtn.disabled = false;
                input.disabled = false;
                input.focus();
                autoResizeTextarea();

                setTimeout(() => {
                    scrollToBottom();
                }, 100);
            }
        }

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const message = input.value.trim();
            if (!message) return;

            const userMessageId = appendMessage('user', message);
            input.value = '';
            autoResizeTextarea();

            sendBtn.disabled = true;
            input.disabled = true;

            await sendMessage(message, userMessageId);
        });

        resetBtn.addEventListener('click', async () => {
            const ok = confirm('새 대화를 시작할까요?');
            if (!ok) return;

            try {
                const response = await fetch('/conversations', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: userId,
                        title: '새 채팅'
                    })
                });

                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data?.detail || '새 대화방 생성 오류');
                }

                currentConversationId = data.conversation.conversation_id;
                localStorage.setItem('current_conversation_id', currentConversationId);
                location.reload();
            } catch (error) {
                console.error(error);
                alert('새 대화를 시작하지 못했습니다.');
            }
        });

        autoResizeTextarea();