import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import date

# 📬 Gmail-instellingen
afzender = "no.reply.de.lijn.9050@gmail.com"
wachtwoord = "gswa uwtp bjfp beva"
ontvangers = ["christoff.rotty@icloud.com"]  # Voeg meer toe indien gewenst

# 📝 Bericht
onderwerp = f"📊 Huishoudboekje update – {date.today().strftime('%d-%m-%Y')}"
bericht = """
Hoi! 👋

Dit is je automatische huishoudupdate.
Open de app om te zien hoe je ervoor staat met je inkomsten en uitgaven:

🔗 https://borolo.streamlit.app/

Tot snel!
"""

def stuur_gmail():
    msg = MIMEMultipart()
    msg['From'] = afzender
    msg['To'] = ", ".join(ontvangers)
    msg['Subject'] = onderwerp
    msg.attach(MIMEText(bericht, 'plain'))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(afzender, wachtwoord)
        server.sendmail(afzender, ontvangers, msg.as_string())
        server.quit()
        print("✅ Mail verzonden via Gmail!")
    except Exception as e:
        print("❌ Fout bij verzenden:", e)

# 🚀 Start verzending
stuur_gmail()
