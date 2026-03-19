import { create } from 'zustand';
import { persist } from 'zustand/middleware';

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
  setActiveVisaCategory: (cat: string | null) => void;
  setChecklist: (items: ChecklistItem[]) => void;
  setRequirements: (reqs: Requirement[]) => void;
  resetProgress: () => void;
  updateMilestone: (milestoneId: string, status: 'completed' | 'current' | 'pending', autoDetected?: boolean) => void;
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

  setActiveVisaCategory: (cat) => set({ activeVisaCategory: cat }),
  setChecklist: (items) => set({ checklist: items }),
  setRequirements: (reqs) => set({ requirements: reqs }),
  
  resetProgress: () => {
    set({
      checklist: [
        { id: '1', title: 'Eligibility Check', status: 'pending' },
        { id: '2', title: 'Point Calculation (0/6)', status: 'pending' },
        { id: '3', title: 'Document Checklist', status: 'pending' },
        { id: '4', title: 'Embassy Appointment', status: 'pending' },
        { id: '5', title: 'Approval', status: 'pending' }
      ],
      requirements: [
        { id: '1', label: 'Language', value: '-', status: 'info' },
        { id: '2', label: 'Work Experience', value: '-', status: 'info' },
        { id: '3', label: 'Age', value: '-', status: 'info' },
        { id: '4', label: 'Qualifications', value: '-', status: 'info' }
      ]
    });
  },

  updateMilestone: (id, status, autoDetected = true) => {
    set((state) => ({
      checklist: state.checklist.map(item => 
        item.id === id ? { ...item, status, autoDetected } : item
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
          visa_types: useChatStore.getState().activeVisaCategory ? [useChatStore.getState().activeVisaCategory] : undefined
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

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
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
                  set((state) => ({
                    messages: state.messages.map(m => 
                      m.id === botMsgId ? { ...m, sources: parsed.metadata.sources } : m
                    )
                  }));
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
              } catch (e) {
                // Ignore incomplete JSON chunks from split bounds if any
              }
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
    activeVisaCategory: state.activeVisaCategory 
  }),
}));
