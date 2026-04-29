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

const currentChatTitle = document.getElementById('current-chat-title');
const DEFAULT_TOPBAR_TITLE = '청년 정책 XAI 상담 채팅';

const conversationPanel = document.querySelector('.conversation-panel');
const conversationToggleBtn = document.getElementById('conversation-toggle-btn');
const conversationClose = document.getElementById('conversation-close');
const conversationBackdrop = document.getElementById('conversation-backdrop');

const exampleChips = document.querySelectorAll('.example-chip');

const heatmapModal = document.getElementById('heatmap-modal');
const heatmapBackdrop = document.getElementById('heatmap-backdrop');
const heatmapClose = document.getElementById('heatmap-close');
const heatmapContent = document.getElementById('heatmap-content');

const deleteModal = document.getElementById('delete-modal');
const deleteBackdrop = document.getElementById('delete-backdrop');
const deleteClose = document.getElementById('delete-close');
const deleteCancel = document.getElementById('delete-cancel');
const deleteConfirm = document.getElementById('delete-confirm');
const deleteDescription = document.getElementById('delete-description');

const conversationList = document.getElementById('conversation-list');
const newChatBtn = document.getElementById('new-chat-btn');
const introCard = document.querySelector('.intro-card');

const DEFAULT_ASSISTANT_MESSAGE =
    '안녕하세요! 자격증, 월세, 취업, 창업, 면접 준비처럼 궁금한 정책을 말씀해 주세요.\n' +
    '상황을 조금 더 자세히 적어주시면 더 적합한 지원 제도를 정리해드릴게요.';

const userId = localStorage.getItem('chat_user_id') || (() => {
    const id = `user_${crypto.randomUUID()}`;
    localStorage.setItem('chat_user_id', id);
    return id;
})();

let currentConversationId = localStorage.getItem('current_conversation_id') || '';
let editingConversationId = '';
let pendingDeleteConversation = null;

/* -----------------------------
 * Layout / viewport helpers
 * ----------------------------- */
function setAppHeight() {
    const viewportHeight = window.visualViewport
        ? window.visualViewport.height
        : window.innerHeight;

    document.documentElement.style.setProperty('--app-height', `${viewportHeight}px`);
}

function lockOverlayScroll() {
    document.body.classList.add('sidebar-open');
}

function unlockOverlayScrollIfNeeded() {
    const sidebarOpen = sidebar?.classList.contains('is-open');
    const conversationOpen = conversationPanel?.classList.contains('is-open');

    if (!sidebarOpen && !conversationOpen) {
        document.body.classList.remove('sidebar-open');
    }
}

function openSidebar() {
    if (window.innerWidth > 960) return;

    closeConversationPanel();

    sidebar.classList.add('is-open');
    sidebarBackdrop.classList.add('is-open');
    lockOverlayScroll();
    helpBtn?.setAttribute('aria-expanded', 'true');
}

function closeSidebar() {
    sidebar.classList.remove('is-open');
    sidebarBackdrop.classList.remove('is-open');
    helpBtn?.setAttribute('aria-expanded', 'false');
    unlockOverlayScrollIfNeeded();
}

function openConversationPanel() {
    if (window.innerWidth > 960) return;

    closeSidebar();

    conversationPanel.classList.add('is-open');
    conversationBackdrop.classList.add('is-open');
    lockOverlayScroll();
    conversationToggleBtn?.setAttribute('aria-expanded', 'true');
}

function closeConversationPanel() {
    conversationPanel?.classList.remove('is-open');
    conversationBackdrop?.classList.remove('is-open');
    conversationToggleBtn?.setAttribute('aria-expanded', 'false');
    unlockOverlayScrollIfNeeded();
}

function openDeleteModal(conversation) {
    pendingDeleteConversation = conversation;
    deleteDescription.textContent = `"${conversation.title || '새 채팅'}" 대화를 삭제할까요?`;
    deleteModal.hidden = false;
    document.body.classList.add('sidebar-open');
}

function closeDeleteModal() {
    deleteModal.hidden = true;
    pendingDeleteConversation = null;
    unlockOverlayScrollIfNeeded();
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
        closeConversationPanel();
    }
});

helpBtn?.addEventListener('click', openSidebar);
sidebarClose?.addEventListener('click', closeSidebar);
sidebarBackdrop?.addEventListener('click', closeSidebar);

conversationToggleBtn?.addEventListener('click', openConversationPanel);
conversationClose?.addEventListener('click', closeConversationPanel);
conversationBackdrop?.addEventListener('click', closeConversationPanel);

/* -----------------------------
 * Shared utility
 * ----------------------------- */
function escapeHtml(text) {
    return String(text || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function renderMarkdown(text) {
    const rawHtml = marked.parse(text || '', { breaks: true });
    return DOMPurify.sanitize(rawHtml);
}

function getTimeLabel() {
    const now = new Date();
    return now.toLocaleTimeString('ko-KR', {
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatConversationDate(isoString) {
    if (!isoString) return '';
    const date = new Date(isoString);
    if (Number.isNaN(date.getTime())) return '';

    return date.toLocaleString('ko-KR', {
        month: 'numeric',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function autoResizeTextarea() {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
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

function showIntroCard() {
    if (introCard) {
        introCard.style.display = '';
    }
}

function hideIntroCard() {
    if (introCard) {
        introCard.style.display = 'none';
    }
}

function getDisplayConversationTitle(title) {
    const normalized = (title || '').trim();

    if (!normalized || normalized === '새 채팅') {
        return DEFAULT_TOPBAR_TITLE;
    }

    return normalized;
}

function setCurrentChatTitle(title) {
    if (!currentChatTitle) return;
    currentChatTitle.textContent = getDisplayConversationTitle(title);
}

function applyCurrentConversationTitleFromConversation(conversation) {
    setCurrentChatTitle(conversation?.title);
}

function applyCurrentConversationTitleFromList(conversations) {
    const current = (conversations || []).find(
        (item) => item.conversation_id === currentConversationId
    );
    setCurrentChatTitle(current?.title);
}

/* -----------------------------
 * API helpers
 * ----------------------------- */
async function fetchJson(url, options = {}, fallbackMessage = '요청 처리 중 오류가 발생했습니다.') {
    const response = await fetch(url, options);
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data?.detail || fallbackMessage);
    }

    return data;
}

async function createConversation(title = '새 채팅') {
    const data = await fetchJson(
        '/conversations',
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: userId,
                title
            })
        },
        '새 대화방 생성 오류'
    );

    return data.conversation;
}

async function fetchConversationList() {
    const data = await fetchJson(
        `/conversations?user_id=${encodeURIComponent(userId)}`,
        {},
        '대화 목록 조회 오류'
    );

    return data.conversations || [];
}

async function fetchConversationDetail(conversationId) {
    const data = await fetchJson(
        `/conversations/${encodeURIComponent(conversationId)}?user_id=${encodeURIComponent(userId)}`,
        {},
        '대화방 조회 오류'
    );

    return data.conversation;
}

async function updateConversationTitleApi(conversationId, title) {
    const data = await fetchJson(
        `/conversations/${encodeURIComponent(conversationId)}/title`,
        {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: userId,
                title
            })
        },
        '대화 제목 수정 오류'
    );

    return data.conversation;
}

async function deleteConversationApi(conversationId) {
    return fetchJson(
        `/conversations/${encodeURIComponent(conversationId)}?user_id=${encodeURIComponent(userId)}`,
        {
            method: 'DELETE'
        },
        '대화 삭제 오류'
    );
}

async function callChatAPI(message) {
    const conversationId = await ensureConversationReady();

    return fetchJson(
        '/chat',
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: userId,
                conversation_id: conversationId,
                message
            })
        },
        '채팅 응답 오류'
    );
}

async function callCaptumAPI(message) {
    const conversationId = await ensureConversationReady();

    return fetchJson(
        '/captum',
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: userId,
                conversation_id: conversationId,
                message
            })
        },
        'Captum 응답 오류'
    );
}

/* -----------------------------
 * Conversation lifecycle
 * ----------------------------- */
async function ensureConversationReady() {
    if (currentConversationId) {
        return currentConversationId;
    }

    const conversation = await createConversation('새 채팅');
    currentConversationId = conversation.conversation_id;
    localStorage.setItem('current_conversation_id', currentConversationId);
    return currentConversationId;
}

async function refreshConversationList() {
    const conversations = await fetchConversationList();
    renderConversationList(conversations);
    applyCurrentConversationTitleFromList(conversations);
}

async function createAndSwitchConversation() {
    const conversation = await createConversation('새 채팅');
    currentConversationId = conversation.conversation_id;
    localStorage.setItem('current_conversation_id', currentConversationId);

    await loadCurrentConversation();
    await refreshConversationList();

    if (window.innerWidth <= 960) {
        closeConversationPanel();
    }
}

async function deleteConversationAndRecover(conversationId) {
    await deleteConversationApi(conversationId);

    if (currentConversationId === conversationId) {
        currentConversationId = '';
        localStorage.removeItem('current_conversation_id');

        const conversations = await fetchConversationList();

        if (conversations.length > 0) {
            currentConversationId = conversations[0].conversation_id;
            localStorage.setItem('current_conversation_id', currentConversationId);
        } else {
            const newConversation = await createConversation('새 채팅');
            currentConversationId = newConversation.conversation_id;
            localStorage.setItem('current_conversation_id', currentConversationId);
        }

        await loadCurrentConversation();
    }

    await refreshConversationList();
}

/* -----------------------------
 * Conversation list rendering
 * ----------------------------- */
function renderConversationList(conversations) {
    conversationList.innerHTML = '';

    if (!Array.isArray(conversations) || conversations.length === 0) {
        conversationList.innerHTML = `
            <div class="conversation-empty">
                저장된 대화가 여기에 표시됩니다.
            </div>
        `;
        return;
    }

    conversations.forEach((conv) => {
        const item = document.createElement('div');
        item.className = `conversation-item${conv.conversation_id === currentConversationId ? ' is-active' : ''}`;

        const isEditing = editingConversationId === conv.conversation_id;

        if (isEditing) {
            const editWrap = document.createElement('div');
            editWrap.className = 'conversation-item__edit';

            const inputEl = document.createElement('input');
            inputEl.type = 'text';
            inputEl.className = 'conversation-edit-input';
            inputEl.value = conv.title || '새 채팅';
            inputEl.maxLength = 60;

            const actions = document.createElement('div');
            actions.className = 'conversation-edit-actions';

            const cancelBtn = document.createElement('button');
            cancelBtn.type = 'button';
            cancelBtn.className = 'conversation-inline-btn';
            cancelBtn.textContent = '취소';

            cancelBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                editingConversationId = '';
                refreshConversationList();
            });

            const saveBtn = document.createElement('button');
            saveBtn.type = 'button';
            saveBtn.className = 'conversation-inline-btn conversation-inline-btn--save';
            saveBtn.textContent = '저장';

            const submitEdit = async () => {
                const trimmed = inputEl.value.trim();
                if (!trimmed) {
                    alert('제목은 비워둘 수 없습니다.');
                    inputEl.focus();
                    return;
                }

                try {
                    await updateConversationTitleApi(conv.conversation_id, trimmed);
                    editingConversationId = '';
                    await refreshConversationList();
                } catch (error) {
                    console.error(error);
                    alert('제목을 수정하지 못했습니다.');
                }
            };

            saveBtn.addEventListener('click', async (e) => {
                e.stopPropagation();
                await submitEdit();
            });

            inputEl.addEventListener('keydown', async (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    await submitEdit();
                }
                if (e.key === 'Escape') {
                    e.preventDefault();
                    editingConversationId = '';
                    await refreshConversationList();
                }
            });

            actions.appendChild(cancelBtn);
            actions.appendChild(saveBtn);

            editWrap.appendChild(inputEl);
            editWrap.appendChild(actions);

            item.appendChild(editWrap);
            conversationList.appendChild(item);

            setTimeout(() => {
                inputEl.focus();
                inputEl.select();
            }, 0);

            return;
        }

        const contentBtn = document.createElement('button');
        contentBtn.type = 'button';
        contentBtn.className = 'conversation-item__content';
        contentBtn.style.all = 'unset';
        contentBtn.style.cursor = 'pointer';
        contentBtn.style.display = 'block';
        contentBtn.style.minWidth = '0';
        contentBtn.style.flex = '1 1 auto';

        contentBtn.innerHTML = `
            <div class="conversation-item__title">${escapeHtml(conv.title || '새 채팅')}</div>
            <div class="conversation-item__meta">${formatConversationDate(conv.updated_at)}</div>
        `;

        contentBtn.addEventListener('click', async () => {
            if (currentConversationId === conv.conversation_id) {
                if (window.innerWidth <= 960) {
                    closeConversationPanel();
                }
                return;
            }

            currentConversationId = conv.conversation_id;
            localStorage.setItem('current_conversation_id', currentConversationId);

            await loadCurrentConversation();
            await refreshConversationList();

            if (window.innerWidth <= 960) {
                closeConversationPanel();
            }
        });

        const actions = document.createElement('div');
        actions.className = 'conversation-item__actions';

        const editBtn = document.createElement('button');
        editBtn.type = 'button';
        editBtn.className = 'conversation-action-btn';
        editBtn.setAttribute('aria-label', '제목 수정');
        editBtn.innerHTML = '<i class="fa-solid fa-pen"></i>';

        editBtn.addEventListener('click', async (e) => {
            e.stopPropagation();
            editingConversationId = conv.conversation_id;
            await refreshConversationList();
        });

        const deleteBtn = document.createElement('button');
        deleteBtn.type = 'button';
        deleteBtn.className = 'conversation-action-btn conversation-action-btn--danger';
        deleteBtn.setAttribute('aria-label', '대화 삭제');
        deleteBtn.innerHTML = '<i class="fa-solid fa-trash"></i>';

        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            openDeleteModal(conv);
        });

        actions.appendChild(editBtn);
        actions.appendChild(deleteBtn);

        item.appendChild(contentBtn);
        item.appendChild(actions);

        conversationList.appendChild(item);
    });
}

/* -----------------------------
 * Message rendering
 * ----------------------------- */
function createTypingIndicator() {
    const wrap = document.createElement('div');
    wrap.className = 'typing';
    wrap.innerHTML = '<span></span><span></span><span></span>';
    return wrap;
}

function appendMessage(sender, text, isLoading = false) {
    const id = `msg-${Date.now()}-${Math.floor(Math.random() * 10000)}`;
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

function clearChatMessages() {
    messages.innerHTML = '';
}

function appendStoredMessage(role, text) {
    const id = appendMessage(role, '', false);
    const row = document.getElementById(id);
    if (!row) return;

    const bubble = row.querySelector('.bubble');
    if (!bubble) return;

    bubble.innerHTML = '';

    if (role === 'assistant') {
        const content = document.createElement('div');
        content.className = 'markdown-body';
        content.innerHTML = renderMarkdown(text || '');
        bubble.appendChild(content);
    } else {
        bubble.textContent = text || '';
    }
}

async function loadCurrentConversation() {
    const conversationId = await ensureConversationReady();

    let conversation;
    try {
        conversation = await fetchConversationDetail(conversationId);
        applyCurrentConversationTitleFromConversation(conversation);
    } catch (error) {
        console.warn('대화방 로드 실패, 새 대화방으로 복구합니다:', error);
        localStorage.removeItem('current_conversation_id');
        currentConversationId = '';
        const recoveredConversationId = await ensureConversationReady();
        conversation = await fetchConversationDetail(recoveredConversationId);
    }

    clearChatMessages();

    const messagesData = Array.isArray(conversation.messages) ? conversation.messages : [];

    if (messagesData.length === 0) {
        showIntroCard();
        appendStoredMessage('assistant', DEFAULT_ASSISTANT_MESSAGE);
        return;
    }

    hideIntroCard();

    messagesData.forEach((msg) => {
        const sender = msg.role === 'assistant' ? 'assistant' : 'user';
        appendStoredMessage(sender, msg.content || '');
    });

    scrollToBottom();
}

/* -----------------------------
 * Captum / heatmap rendering
 * ----------------------------- */
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
        .filter((w) => w && typeof w.start === 'number' && typeof w.end === 'number')
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

    let html = '<div class="user-highlight">';
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

    html += '</div>';
    html += '<div class="user-highlight-note">Captum 기준으로 입력 질문에서 상대적으로 중요한 단어를 색으로 표시했습니다.</div>';

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

function getHeatColor(score) {
    const alpha = Math.max(0.12, Math.min(score, 1) * 0.75);
    return `rgba(37, 99, 235, ${alpha})`;
}

function buildHeatStripHtml(wordScores) {
    return `
        <div class="heat-strip">
            ${wordScores.map((item) => `
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
            ${sorted.map((item) => `
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

    const wordScores = captumResult.word_scores.filter((item) => item && item.word);

    return `
        <div class="heatmap-panel">
            <div class="heatmap-question">${escapeHtml(originalMessage)}</div>
            ${buildHeatStripHtml(wordScores)}
            ${buildBarChartHtml(wordScores)}
        </div>
    `;
}

function openHeatmapModal(originalMessage, captumResult) {
    closeDeleteModal();
    heatmapContent.innerHTML = buildHeatmapModalHtml(originalMessage, captumResult);
    heatmapModal.hidden = false;
    document.body.classList.add('sidebar-open');
}

function closeHeatmapModal() {
    heatmapModal.hidden = true;
    heatmapContent.innerHTML = '';
    unlockOverlayScrollIfNeeded();
}

heatmapClose?.addEventListener('click', closeHeatmapModal);
heatmapBackdrop?.addEventListener('click', closeHeatmapModal);
deleteClose?.addEventListener('click', closeDeleteModal);
deleteBackdrop?.addEventListener('click', closeDeleteModal);
deleteCancel?.addEventListener('click', closeDeleteModal);

deleteConfirm?.addEventListener('click', async () => {
    if (!pendingDeleteConversation) return;

    try {
        await deleteConversationAndRecover(pendingDeleteConversation.conversation_id);

        if (window.innerWidth <= 960) {
            closeConversationPanel();
        }

        closeDeleteModal();
    } catch (error) {
        console.error(error);
        alert('대화를 삭제하지 못했습니다.');
    }
});

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

/* -----------------------------
 * Chat sending
 * ----------------------------- */
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

        await refreshConversationList();

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

/* -----------------------------
 * Form / button events
 * ----------------------------- */
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

    if (!deleteModal.hidden) {
    closeDeleteModal();
    return;
}
});

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const message = input.value.trim();
    if (!message) return;

    const userMessageId = appendMessage('user', message);
    input.value = '';
    autoResizeTextarea();
    hideIntroCard();

    sendBtn.disabled = true;
    input.disabled = true;

    await sendMessage(message, userMessageId);
});

newChatBtn?.addEventListener('click', async () => {
    try {
        await createAndSwitchConversation();
    } catch (error) {
        console.error(error);
        alert('새 대화를 시작하지 못했습니다.');
    }
});

resetBtn.addEventListener('click', async () => {
    const ok = confirm('새 대화를 시작할까요?');
    if (!ok) return;

    try {
        await createAndSwitchConversation();
    } catch (error) {
        console.error(error);
        alert('새 대화를 시작하지 못했습니다.');
    }
});

/* -----------------------------
 * Example chips
 * ----------------------------- */
exampleChips.forEach((chip) => {
    chip.addEventListener('click', () => {
        const chipText = chip.textContent.trim();
        let template = '';

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

        input.value = template;
        autoResizeTextarea();
        input.focus();

        if (window.innerWidth <= 960) {
            closeSidebar();
            closeConversationPanel();
        }

        setTimeout(() => {
            scrollToBottom(true);
        }, 120);
    });
});

/* -----------------------------
 * Global key handling
 * ----------------------------- */
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        if (!heatmapModal.hidden) {
            closeHeatmapModal();
            return;
        }

        if (sidebar.classList.contains('is-open')) {
            closeSidebar();
        }

        if (conversationPanel.classList.contains('is-open')) {
            closeConversationPanel();
        }
    }
});

/* -----------------------------
 * Bootstrap
 * ----------------------------- */
async function bootstrapApp() {
    setCurrentChatTitle(DEFAULT_TOPBAR_TITLE);
    try {
        await ensureConversationReady();
        await refreshConversationList();
        await loadCurrentConversation();
    } catch (error) {
        console.error(error);
    } finally {
        autoResizeTextarea();
    }
}

bootstrapApp();