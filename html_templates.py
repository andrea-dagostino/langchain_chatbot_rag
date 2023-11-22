css = """<style>
/* Container & General Layout */
.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f4f4f4;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.chat-message {
    border-radius: 20px;
    margin-bottom: 15px;
}

/* Bot Message Styling */
.chat-message.bot {
    background-color: #e6e6ff;
}

.chat-message.bot .avatar img {
    border: 2px solid #3333ff;
}

.chat-message.bot .message {
    background-color: #ccf;
    color: #333;
    border: 1px solid #3333ff;
}

/* User Message Styling */
.chat-message.user {
    background-color: #fff0f0;
}

.chat-message.user .avatar img {
    border: 2px solid #ff3333;
}

.chat-message.user .message {
    background-color: #fcc;
    color: #333;
    border: 1px solid #ff3333;
}

/* Responsiveness and Media Queries */
@media (max-width: 600px) {
    .container {
        padding: 10px;
    }

    .chat-message .avatar img {
        max-width: 50px;
    }
}

/* Typography and Spacing */
.message {
    font-family: 'Arial', sans-serif;
    line-height: 1.5;
    padding: 10px 15px;
}

</style>"""

bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
"""
