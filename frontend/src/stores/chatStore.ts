import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { useSettingsStore } from './settingsStore';
import { useToastStore } from './toastStore';
import { translations } from '../translations';

export interface Source {
  url: string;
  title: string;
  authority: string;
}

export interface ChecklistItem {
  id: string;
  title: string;
  status: 'completed' | 'current' | 'pending';
  subtitle?: string;
  autoDetected?: boolean;
}

export interface Requirement {
  id: string;
  label: string;
  value: string;
  status: 'required' | 'info' | 'warning';
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  status?: string;
}

interface ChatState {
  messages: Message[];
  isLoading: boolean;
  activeVisaCategory: string | null;
  checklist: ChecklistItem[];
  requirements: Requirement[];
  recentSources: Source[];
  pinnedSources: Source[];
  setActiveVisaCategory: (cat: string | null) => void;
  setChecklist: (items: ChecklistItem[]) => void;
  setRequirements: (reqs: Requirement[]) => void;
  resetProgress: () => void;
  updateMilestone: (milestoneId: string, status: 'completed' | 'current' | 'pending', autoDetected?: boolean) => void;
  updateRequirement: (id: string, value: string, status: 'required' | 'info' | 'warning') => void;
  sendMessage: (content: string) => Promise<void>;
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
  messages: [
    {
      id: 'welcome',
      role: 'assistant',
      content: 'Hello! I am **VisaPilot AI**. I can help you understand German visa regulations, the *Chancenkarte*, and official requirements.'
    }
  ],
  isLoading: false,
  activeVisaCategory: null,
  
  checklist: [
    { id: '1', title: 'Eligibility Check', status: 'completed' },
    { id: '2', title: 'Point Calculation (0/6)', status: 'current', subtitle: 'Current' },
    { id: '3', title: 'Document Checklist', status: 'pending' },
    { id: '4', title: 'Embassy Appointment', status: 'pending' },
    { id: '5', title: 'Approval', status: 'pending' }
  ],
  requirements: [
    { id: '1', label: 'Language', value: '(B1 German)', status: 'required' },
    { id: '2', label: 'Work Experience', value: '2+ Years', status: 'info' },
    { id: '3', label: 'Age', value: '< 35/40', status: 'info' },
    { id: '4', label: 'Qualifications', value: '(degree)', status: 'warning' }
  ],
  
  pinnedSources: [
    { title: "Federal Ministry (BMI)", url: "https://www.bmi.bund.de", authority: "official" },
    { title: "Make it in Germany", url: "https://www.make-it-in-germany.com", authority: "official" },
    { title: "Foreign Office", url: "https://www.auswaertiges-amt.de/en/visa-service", authority: "official" },
    { title: "Consular Portal", url: "https://digital.diplo.de/visa", authority: "official" }
  ],
  recentSources: [],

  setActiveVisaCategory: (cat) => {
    set({ activeVisaCategory: cat });
    get().resetProgress();
  },
  setChecklist: (items) => set({ checklist: items }),
  setRequirements: (reqs) => set({ requirements: reqs }),
  
  resetProgress: () => {
    const cat = get().activeVisaCategory;
    
    let checklist: ChecklistItem[] = [];
    let requirements: Requirement[] = [];

    if (cat === 'work_visa') {
      checklist = [
        { id: '1', title: 'Professional Qualifications', status: 'pending' },
        { id: '2', title: 'Full-time Contract', status: 'pending' },
        { id: '3', title: 'Document Checklist', status: 'pending' }
      ];
      requirements = [
        { id: '1', label: 'Qualifications', value: '-', status: 'info' },
        { id: '2', label: 'Labor Conditions', value: '-', status: 'info' },
        { id: '3', label: '45+ Age Clause', value: '-', status: 'info' },
        { id: '4', label: 'Language (Flexible)', value: '-', status: 'info' }
      ];
    } else if (cat === 'blue_card') {
      checklist = [
        { id: '1', title: 'Degree Recognition', status: 'pending' },
        { id: '2', title: 'High Salary Threshold', status: 'pending' },
        { id: '3', title: 'Document Checklist', status: 'pending' }
      ];
      requirements = [
        { id: '1', label: 'Academic & Professional Qualifications', value: '-', status: 'info' },
        { id: '2', label: 'German Full-time Contract', value: '-', status: 'info' },
        { id: '3', label: 'PR Bonus (Language)', value: '-', status: 'info' }
      ];
    } else if (cat === 'student_visa') {
      checklist = [
        { id: '1', title: 'Finance & Language', status: 'pending' },
        { id: '2', title: 'University Admission', status: 'pending' },
        { id: '3', title: 'Document Checklist', status: 'pending' }
      ];
      requirements = [
        { id: '1', label: 'Financial Proof', value: '-', status: 'info' },
        { id: '2', label: 'Language Ability', value: '-', status: 'info' },
        { id: '3', label: 'Health Insurance', value: '-', status: 'info' },
        { id: '4', label: 'Pre-study Qualifications', value: '-', status: 'info' }
      ];
    } else {
      checklist = [
        { id: '1', title: 'Basic Thresholds', status: 'pending' },
        { id: '2', title: 'Points Calculation', status: 'pending' },
        { id: '3', title: 'Document Checklist', status: 'pending' }
      ];
      requirements = [
        { id: 'header1', label: 'Mandatory Thresholds', value: '', status: 'info' },
        { id: '1-1', label: 'Financial Proof', value: '-', status: 'info' },
        { id: '1-2', label: 'Language', value: '-', status: 'info' },
        { id: '1-3', label: 'Qualifications', value: '-', status: 'info' },
        { id: 'header2', label: 'Points Items (Target 6+)', value: '', status: 'info' },
        { id: '2-1', label: 'Language', value: '-', status: 'info' },
        { id: '2-2', label: 'Work Experience', value: '-', status: 'info' },
        { id: '2-3', label: 'Age', value: '-', status: 'info' },
        { id: '2-4', label: 'Qualifications', value: '-', status: 'info' },
        { id: '2-5', label: 'German Residency (6mo+)', value: '-', status: 'info' },
        { id: '2-6', label: 'Partner Bonus', value: '-', status: 'info' }
      ];
    }

    set({ checklist, requirements });
  },

  updateMilestone: (id, status, autoDetected = true) => {
    set((state) => {
      const item = state.checklist.find(i => i.id === id);
      const isStatusChanged = item && item.status !== status;
      
      // Fire notification if status changed and feature is enabled
      if (isStatusChanged && (status === 'current' || status === 'completed')) {
        const settings = useSettingsStore.getState();
        if (settings.notificationsEnabled) {
          const t = translations[settings.language as keyof typeof translations] || translations.en;
          
          let translatedTitle = item.title;
          const lower = item.title.toLowerCase();
          if (lower.includes('eligibility path')) translatedTitle = t.criteria.eligibilityPath || item.title;
          else if (lower.includes('eligibility check')) translatedTitle = t.criteria.eligibility || item.title;
          else if (lower.includes('basic thresholds')) translatedTitle = t.criteria.basicThresholds || item.title;
          else if (lower.includes('salary threshold')) translatedTitle = t.criteria.salaryThreshold || item.title;
          else if (lower.includes('points calculation') || lower.includes('points requirements')) translatedTitle = t.pointsReq || item.title;
          else if (lower.includes('university admission') || lower.includes('zulassung')) translatedTitle = t.criteria.uniAdmission || item.title;
          else if (lower.includes('job offer') || lower.includes('full-time contract') || lower.includes('german full-time contract')) translatedTitle = t.criteria.fullTimeContract || item.title;
          else if (lower.includes('salary check')) translatedTitle = t.salaryCheck || item.title;
          else if (lower.includes('document')) translatedTitle = t.docChecklist || item.title;
          else if (lower.includes('embassy') || lower.includes('appointment')) translatedTitle = t.embassyAppt || item.title;
          else if (lower.includes('approval') || lower.includes('final step')) translatedTitle = t.approval || item.title;

          useToastStore.getState().addToast({
            title: t.milestoneReached || 'Milestone Reached',
            message: `${translatedTitle}: ${t.autoDetectedDesc || 'AI has updated your checklist'}`,
            type: 'success'
          });
        }
      }

      return {
        checklist: state.checklist.map(i => 
          i.id === id ? { ...i, status, autoDetected } : i
        )
      };
    });
  },

  updateRequirement: (id, value, status) => {
    set((state) => ({
      requirements: state.requirements.map(req => 
        req.id === id ? { ...req, value, status } : req
      )
    }));
  },
  
  sendMessage: async (content: string) => {
    if (!content.trim()) return;
    
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content };
    const botMsgId = (Date.now() + 1).toString();
    
    set((state) => ({ 
      messages: [...state.messages, userMsg, { id: botMsgId, role: 'assistant', content: '' }],
      isLoading: true 
    }));

    try {
      const response = await fetch('/query/ask/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': import.meta.env.VITE_API_KEY || ''
        },
        body: JSON.stringify({ 
          query: content,
          language: useSettingsStore.getState().language,
          visa_type: useChatStore.getState().activeVisaCategory || undefined,
          requirements: get().requirements
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      if (!response.body) throw new Error("No response body from server");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let lineBuffer = '';

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          lineBuffer += decoder.decode(value, { stream: true });
          const lines = lineBuffer.split('\n');
          lineBuffer = lines.pop() || '';
          
          for (const line of lines) {
            const trimmedLine = line.trim();
            if (!trimmedLine || !trimmedLine.startsWith('data: ')) continue;
            
            const data = trimmedLine.slice(6);
            if (data === '[DONE]') continue;
              
              try {
                const parsed = JSON.parse(data);
                
                // Content Chunk
                if (parsed.choices?.[0]?.delta?.content) {
                  set((state) => ({
                    messages: state.messages.map(m => 
                      m.id === botMsgId ? { ...m, content: m.content + parsed.choices[0].delta.content } : m
                    )
                  }));
                }
                
                // Custom Metadata Chunk (Sources / Status)
                if (parsed.metadata?.sources) {
                  const newSources = parsed.metadata.sources as Source[];
                  
                  set((state) => {
                    // Harvest unique sources for the sidebar
                    const currentRecent = [...state.recentSources];
                    newSources.forEach(s => {
                      if (!currentRecent.find(existing => existing.url === s.url)) {
                        currentRecent.unshift(s);
                      }
                    });
                    
                    return {
                      messages: state.messages.map(m => 
                        m.id === botMsgId ? { ...m, sources: newSources } : m
                      ),
                      recentSources: currentRecent.slice(0, 5) // Keep last 5 unique ones
                    };
                  });
                }
                
                if (parsed.metadata?.status) {
                  set((state) => ({
                    messages: state.messages.map(m => 
                      m.id === botMsgId ? { ...m, status: parsed.metadata.status } : m
                    )
                  }));
                }

                if (parsed.metadata?.achieved_milestone) {
                  const { id, status } = parsed.metadata.achieved_milestone;
                  get().updateMilestone(id, status, true);
                }

                if (parsed.metadata?.updated_requirement) {
                  const { id, value, status } = parsed.metadata.updated_requirement;
                  get().updateRequirement(id, value, status);
                }
              } catch (e) {
                // Ignore incomplete JSON chunks from split bounds if any
              }
          }
        }
      }
    } catch (err) {
      const error = err as Error;
      console.error("Chat streaming error:", error);
      let errorMsg = "_Sorry, an error occurred while streaming the response._";
      
      if (error.message?.includes("429") || error.message?.includes("quota")) {
        errorMsg = "\n\n> ⚠️ **API Rate Limit Reached**\n> You have exceeded the free tier quota (24 RPM). Please wait about 30-60 seconds before trying again.";
      } else if (error.message) {
        errorMsg = `\n\n_Error: ${error.message}_`;
      }
      
      set((state) => ({
        messages: state.messages.map(m => 
          m.id === botMsgId ? { ...m, content: m.content + errorMsg } : m
        )
      }));
    } finally {
      set({ isLoading: false });
    }
  }
}), {
  name: 'visa-rag-chat-storage',
  partialize: (state) => ({ 
    checklist: state.checklist, 
    requirements: state.requirements,
    recentSources: state.recentSources,
    activeVisaCategory: state.activeVisaCategory 
  }),
}));
