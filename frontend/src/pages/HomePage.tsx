import { ChatArea } from '../components/ChatArea';
import { InsightsPanel } from '../components/InsightsPanel';

export function HomePage() {
  return (
    <>
      <ChatArea className="flex-1 min-w-[500px]" />
      <InsightsPanel className="w-[25%] min-w-[320px]" />
    </>
  );
}
