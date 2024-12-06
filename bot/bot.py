from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

BOT_TOKEN = '7810720783:AAGu08mMx04xSMeiX-DnsLwFeLyZt9IaF54'

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Respond to the /start command."""
    await update.message.reply_text("Hello! Send me an audio file, and I'll respond with a dummy message.")

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming audio files."""
    # Extract file information
    audio_file = update.message.audio
    if not audio_file:
        await update.message.reply_text("This doesn't seem like an audio file. Please send a proper audio file.")
        return
    
    # Dummy response
    await update.message.reply_text(f"Received your audio file titled: {audio_file.file_name or 'unknown'}. Processing... (dummy response)")

def main():
    """Run the bot."""
    # Initialize the bot
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    # Command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.AUDIO, handle_audio))

    # Start polling
    application.run_polling()

if __name__ == "__main__":
    main()
