import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import date

# 📨 Instellingen
afzender = "bombaert-rotty@outlook.com"
wachtwoord = "dipnlrtqoduhtret"
ontvangers = ["christof.rotty@icloud.com", "partner@outlook.com"]

onderwerp = f"💰 Huishoudupdate – {date.today().strftime('%d-%m-%Y')}"
bericht = """
Hoi! 👋

Dit is je automatische huishoudupdate.
Open de app om te zien hoe je ervoor staat met je inkomsten en uitgaven:

🔗 https://borolo.streamlit.app/

Tot snel!
"""

def stuur_outlook_email():
    msg = MIMEMultipart()
    msg['From'] = afzender
    msg['To'] = ", ".join(ontvangers)
    msg['Subject'] = onderwerp
    msg.attach(MIMEText(bericht, 'plain'))

    try:
        server = smtplib.SMTP('smtp.office365.com', 587)
        server.starttls()
        server.login(afzender, wachtwoord)
        server.sendmail(afzender, ontvangers, msg.as_string())
        server.quit()
        print("✅ Mail verzonden.")
    except Exception as e:
        print(f"❌ Fout bij verzenden: {e}")

# 🔁 Verstuur nu
stuur_outlook_email()

