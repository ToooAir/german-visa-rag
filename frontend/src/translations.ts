import { useSettingsStore, Language } from './stores/settingsStore';

export const translations = {
  en: {
    // Sidebar
    home: 'Home',
    library: 'Knowledge Base',
    settings: 'Settings',
    recentSources: 'Recent Sources',
    visaCategories: 'Visa Categories',
    
    // Settings
    appearance: 'Appearance',
    themePreference: 'Theme Preference',
    langRegion: 'Language & Region',
    chatLang: 'Chat Language',
    notifications: 'Notifications',
    emailAlerts: 'Email Alerts for Visa Updates',
    smartAlerts: 'Smart Progress Alerts',
    smartAlertsDesc: 'Get notified when AI updates your milestone.',
    milestoneReached: 'Milestone Reached!',
    milestoneAutoUpdated: 'AI has detected progress and automatically updated your checklist.',
    dark: 'Dark Mode',
    light: 'Light Mode',
    system: 'System Default',
    
    // Chat
    welcome: 'Hello! I am **VisaPilot AI**. I can help you understand German visa regulations, the *Chancenkarte*, and official requirements.',
    chatHeader: 'Chat',
    askPlaceholder: 'Ask VisaPilot AI...',
    showcaseExamples: 'Showcase Examples',
    you: 'You',
    
    // Insights
    progress: 'Progress',
    reqSummary: 'Requirement Summary',
    pointsReq: 'Points Required',
    mainCriteria: 'Main Criteria',
    met: 'met',
    analyzing: 'Analyzing intent & parameters...',
    retrieving: 'Querying official vector database...',
    extracting: 'Extracting legal requirements...',
    synthesizing: 'Synthesizing answer...',
    docChecklist: 'Document Checklist',
    embassyAppt: 'Embassy Appointment',
    approval: 'Approval',
    eligibilityDesc: 'Based on your current profile, you meet the primary criteria but need a few more points for the Chancenkarte.',
    resetProgress: 'Reset Progress',
    autoDetected: 'Auto-detected by AI',
    autoDetectedDesc: 'This milestone was automatically identified from your conversation history.',
    pointsMet: 'points met',
    pointsRequired: 'Points Required',
    
    // Visa Categories
    chancenkarte: 'Chancenkarte',
    skilledWorker: 'Skilled Worker',
    blueCard: 'Blue Card',
    studyVisa: 'Study Visa',

    // Showcase
    showcase: {
      pointCalc: { title: 'Point Calculation', query: 'Am I eligible for the Chancenkarte with 5 years experience and B1 German?' },
      blueCardSalary: { title: 'Blue Card Salary', query: 'What is the new minimum salary for the EU Blue Card in 2024?' },
      studentWork: { title: 'Student Work Rights', query: 'Can international students work part-time? Explain the latest updates.' }
    },

    // Requirements
    criteria: {
      language: 'Language',
      experience: 'Work Experience',
      age: 'Age',
      qualifications: 'Qualifications',
      eligibility: 'Eligibility Check'
    }
  },
  de: {
    // Sidebar
    home: 'Startseite',
    library: 'Wissensdatenbank',
    settings: 'Einstellungen',
    recentSources: 'Aktuelle Quellen',
    visaCategories: 'Visum-Kategorien',
    
    // Settings
    appearance: 'Erscheinungsbild',
    themePreference: 'Design-Einstellung',
    langRegion: 'Sprache & Region',
    chatLang: 'Chat-Sprache',
    notifications: 'Benachrichtigungen',
    emailAlerts: 'E-Mail-Benachrichtigungen für Visum-Updates',
    smartAlerts: 'Intelligente Fortschrittsalarme',
    smartAlertsDesc: 'Werden Sie benachrichtigt, wenn die KI Ihren Meilenstein aktualisiert.',
    milestoneReached: 'Meilenstein Erreicht!',
    milestoneAutoUpdated: 'KI hat Ihre Fortschritts-Checkliste automatisch aktualisiert.',
    dark: 'Dunkler Modus',
    light: 'Heller Modus',
    system: 'Systemstandard',
    
    // Chat
    welcome: 'Hallo! Ich bin **VisaPilot AI**. Ich kann Ihnen helfen, deutsche Visumbestimmungen, die *Chancenkarte* und offizielle Anforderungen zu verstehen.',
    chatHeader: 'Chat',
    askPlaceholder: 'Fragen Sie VisaPilot AI...',
    showcaseExamples: 'Beispiele ansehen',
    you: 'Sie',
    
    // Insights
    progress: 'Fortschritt',
    reqSummary: 'Anforderungszusammenfassung',
    pointsReq: 'Erforderliche Punkte',
    mainCriteria: 'Hauptkriterien',
    met: 'erfüllt',
    analyzing: 'Intent & Parameter werden analysiert...',
    retrieving: 'Offizielle Vektordatenbank wird abgefragt...',
    extracting: 'Rechtliche Anforderungen werden extrahiert...',
    synthesizing: 'Antwort wird synthetisiert...',
    docChecklist: 'Dokumenten-Checkliste',
    embassyAppt: 'Botschaftstermin',
    approval: 'Genehmigung',
    eligibilityDesc: 'Basierend auf Ihrem aktuellen Profil erfüllen Sie die Hauptkriterien, benötigen aber noch einige Punkte für die Chancenkarte.',
    resetProgress: 'Fortschritt zurücksetzen',
    autoDetected: 'KI-automatisch erkannt',
    autoDetectedDesc: 'Dieser Meilenstein wurde automatisch aus Ihrem Gesprächsverlauf identifiziert.',
    pointsMet: 'Punkte erfüllt',
    pointsRequired: 'Erforderliche Punkte',

    // Visa Categories
    chancenkarte: 'Chancenkarte',
    skilledWorker: 'Fachkraft',
    blueCard: 'Blaue Karte EU',
    studyVisa: 'Visum zum Studium',

    // Showcase
    showcase: {
      pointCalc: { title: 'Punkteberechnung', query: 'Bin ich mit 5 Jahren Erfahrung und B1 Deutsch für die Chancenkarte berechtigt?' },
      blueCardSalary: { title: 'Gehalt Blaue Karte', query: 'Was ist das neue Mindestgehalt für die Blaue Karte EU im Jahr 2024?' },
      studentWork: { title: 'Arbeitsrechte für Studenten', query: 'Können internationale Studierende Teilzeit arbeiten? Erklären Sie die neuesten Updates.' }
    },

    // Requirements
    criteria: {
      language: 'Sprache',
      experience: 'Berufserfahrung',
      age: 'Alter',
      qualifications: 'Qualifikationen',
      eligibility: 'Berechtigungsprüfung'
    }
  },
  'zh-TW': {
    // Sidebar
    home: '首頁',
    library: '法規知識庫',
    settings: '設定',
    recentSources: '最近來源',
    visaCategories: '簽證類別',
    
    // Settings
    appearance: '外觀',
    themePreference: '主題偏好',
    langRegion: '語言與區域',
    chatLang: '聊天語言',
    notifications: '通知',
    emailAlerts: '簽證更新電子郵件通知',
    smartAlerts: '智能進度提報',
    smartAlertsDesc: '當 AI 自動更新您的里程碑時接收通知',
    milestoneReached: '🎊 里程碑達成！',
    milestoneAutoUpdated: 'AI 已偵測進度並自動更新了您的查檢表。',
    dark: '深色模式',
    light: '淺色模式',
    system: '系統預設',
    
    // Chat
    welcome: '您好！我是 **VisaPilot AI**。我可以協助您了解德國簽證法規、*機會卡 (Chancenkarte)* 以及官方要求。',
    chatHeader: '聊天',
    askPlaceholder: '詢問 VisaPilot AI...',
    showcaseExamples: '展示範例',
    you: '您',
    
    // Insights
    progress: '進度',
    reqSummary: '要求摘要',
    pointsReq: '所需積分',
    mainCriteria: '主要標準',
    met: '達成',
    analyzing: '正在規劃搜尋參數...',
    retrieving: '正在檢索官方法規資料庫...',
    extracting: '正在提取相關條款與細節...',
    synthesizing: '正在彙整簽證建議與回答...',
    docChecklist: '文件清單',
    embassyAppt: '使館面談預約',
    approval: '核准',
    eligibilityDesc: '根據您目前的個人資料，您已符合主要標準，但還需要更多積分才能獲得機會卡。',
    resetProgress: '重置進度',
    autoDetected: 'AI 自動偵測',
    autoDetectedDesc: '此進度是根據您的對話紀錄自動分析得出的。',
    pointsMet: '積分已達成',
    pointsRequired: '所需積分',

    // Visa Categories
    chancenkarte: '機會卡',
    skilledWorker: '技術人才簽證',
    blueCard: '藍卡 (Blue Card)',
    studyVisa: '就學簽證',

    // Showcase
    showcase: {
      pointCalc: { title: '積分計算', query: '我有 5 年經驗且德語 B1，符合申請機會卡的資格嗎？' },
      blueCardSalary: { title: '藍卡薪資門檻', query: '2024 年歐盟藍卡的新最低薪資要求是多少？' },
      studentWork: { title: '學生工作權利', query: '外籍學生可以兼職工作嗎？請解釋最新的規定。' }
    },

    // Requirements
    criteria: {
      language: '語言能力',
      experience: '工作經驗',
      age: '年齡',
      qualifications: '學經歷資格',
      eligibility: '資格審查'
    }
  }
};

export const useTranslation = () => {
  const language = useSettingsStore((state) => state.language);
  const t = translations[language as Language] || translations.en;
  return { t, language };
};
