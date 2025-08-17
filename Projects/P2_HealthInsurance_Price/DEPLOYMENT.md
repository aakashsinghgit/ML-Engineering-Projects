# 🚀 Deployment Guide - Health Insurance Price Predictor

This guide will help you deploy your Flask app to production using **Render** (free tier) with automated CI/CD via GitHub Actions.

## 🎯 **What We'll Accomplish**

1. ✅ **Free Hosting** on Render (750 hours/month free)
2. ✅ **Automated Deployment** via GitHub Actions
3. ✅ **Production-Ready** Flask app
4. ✅ **Custom Domain** (optional)

---

## 📋 **Prerequisites**

- ✅ Your Flask app runs locally with Docker
- ✅ GitHub repository with your code
- ✅ Render account (free at [render.com](https://render.com))

## 🔍 **Step 0: Verify Configuration (Important!)**

Before deploying, verify your configuration is correct:

```bash
cd Projects/P2_HealthInsurance_Price
python verify_deployment.py
```

This script checks that all deployment files are properly configured for the subfolder deployment.

---

## 🚀 **Step 1: Deploy to Render (Free)**

### **1.1 Create Render Account**
- Go to [render.com](https://render.com)
- Sign up with GitHub (recommended for easy integration)

### **1.2 Connect Your Repository**
- Click "New +" → "Web Service"
- Connect your GitHub repository
- Select the repository: `ML-Engineering-Projects`
- **Important**: Since your repo has multiple projects, Render will use the `render.yaml` file to know which subfolder to deploy

### **1.3 Configure the Service**
```
Name: health-insurance-predictor
Environment: Python 3
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app --bind 0.0.0.0:$PORT
```

### **1.4 Deploy**
- Click "Create Web Service"
- Render will automatically build and deploy your app
- Your app will be available at: `https://your-app-name.onrender.com`

---

## 🔄 **Step 2: Set Up GitHub Actions CI/CD**

### **2.1 Push Your Changes**
```bash
git add .
git commit -m "Add CI/CD and deployment configuration"
git push origin main
```

### **2.2 Verify GitHub Actions**
- Go to your GitHub repository
- Click "Actions" tab
- You should see the CI/CD pipeline running
- ✅ **Tests pass** → ✅ **Deployment ready**

---

## 🧪 **Step 3: Test Your Deployment**

### **3.1 Check Your Live App**
- Visit your Render URL
- Test the health insurance prediction form
- Verify all functionality works

### **3.2 Monitor Logs**
- In Render dashboard, click on your service
- Check "Logs" tab for any errors
- Monitor "Metrics" for performance

---

## 🔧 **Step 4: Custom Domain (Optional)**

### **4.1 Add Custom Domain**
- In Render dashboard, go to "Settings"
- Click "Custom Domains"
- Add your domain (e.g., `predictor.yourdomain.com`)

### **4.2 DNS Configuration**
```
Type: CNAME
Name: predictor
Value: your-app-name.onrender.com
```

---

## 📊 **What Happens on Every Push**

1. **🔄 GitHub Actions Triggers**
   - Runs tests
   - Builds Docker image
   - Verifies functionality

2. **🚀 Render Auto-Deploys**
   - Detects changes via webhook
   - Rebuilds and deploys automatically
   - Zero downtime updates

---

## 🆓 **Free Tier Limits**

| Platform | Free Tier | Your App |
|----------|-----------|----------|
| **Render** | 750 hours/month | ✅ Always on |
| **GitHub Actions** | 2000 minutes/month | ✅ Plenty for CI/CD |
| **Custom Domain** | Unlimited | ✅ Free |

---

## 🐛 **Troubleshooting**

### **Common Issues:**

1. **Build Fails**
   - Check requirements.txt
   - Verify Python version compatibility
   - Check Render logs

2. **Wrong Project Deployed**
   - Verify `render.yaml` has correct `rootDir: Projects/P2_HealthInsurance_Price`
   - Ensure `render.yaml` is in the root of your repository
   - Check that Render is reading the correct configuration file

2. **App Won't Start**
   - Verify PORT environment variable
   - Check gunicorn configuration
   - Review app.py for production settings

3. **GitHub Actions Fail**
   - Check workflow syntax
   - Verify branch names (main vs master)
   - Review test outputs

---

## 🎉 **Success!**

Your health insurance price predictor is now:
- ✅ **Live on the internet** (free!)
- ✅ **Automatically deployed** on every push
- ✅ **Production-ready** with proper error handling
- ✅ **Scalable** and monitored

---

## 🔗 **Next Steps**

1. **Add Monitoring**: Set up alerts for downtime
2. **Performance**: Add caching and optimization
3. **Security**: Add rate limiting and input validation
4. **Analytics**: Track usage and predictions

---

## 📞 **Need Help?**

- **Render Docs**: [docs.render.com](https://docs.render.com)
- **GitHub Actions**: [docs.github.com/en/actions](https://docs.github.com/en/actions)
- **Flask Production**: [flask.palletsprojects.com/en/2.3.x/deploying/](https://flask.palletsprojects.com/en/2.3.x/deploying/)

---

*Your ML Engineering journey continues! 🚀*
