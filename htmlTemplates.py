css = '''
<style>
.chat-message {
    padding: 0.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 10%;
}
.chat-message .avatar img {
  max-width: 48px;
  max-height: 48px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''
                            
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcScoCHK-AZAbTm8fHw8ud0AVt15RgPQgMFC1Q-1opIjIQ&s">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ1VyCFr3rHLultu7DYy5oRiJlAO-eTOSdXLRhBIfhlGQ&s">>
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''