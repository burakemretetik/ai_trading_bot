
import { supabase } from '@/integrations/supabase/client';
import { toast } from 'sonner';

export async function getTrackedStocks() {
  try {
    const { data: userSession } = await supabase.auth.getSession();
    if (!userSession.session) {
      console.error('User not authenticated');
      return [];
    }

    const userId = userSession.session.user.id;

    const { data, error } = await supabase
      .from('tracked_stocks')
      .select('stock_id')
      .eq('user_id', userId);

    if (error) {
      console.error('Error fetching tracked stocks:', error);
      return [];
    }

    return data.map(item => item.stock_id);
  } catch (error) {
    console.error('Error in getTrackedStocks:', error);
    return [];
  }
}

export async function trackStock(stockId: string) {
  try {
    const { data: userSession } = await supabase.auth.getSession();
    if (!userSession.session) {
      console.error('User not authenticated');
      toast.error('Bu işlem için giriş yapmalısınız');
      return false;
    }

    const userId = userSession.session.user.id;

    const { error } = await supabase
      .from('tracked_stocks')
      .insert({ 
        stock_id: stockId,
        user_id: userId 
      });

    if (error) {
      if (error.code === '23505') {
        // Unique violation - stock already tracked
        console.log('Stock already tracked');
      } else {
        console.error('Error tracking stock:', error);
        toast.error('Hisse takip edilemedi');
        return false;
      }
    }
    
    return true;
  } catch (error) {
    console.error('Error in trackStock:', error);
    toast.error('Hisse takip edilemedi');
    return false;
  }
}

export async function untrackStock(stockId: string) {
  try {
    const { data: userSession } = await supabase.auth.getSession();
    if (!userSession.session) {
      console.error('User not authenticated');
      toast.error('Bu işlem için giriş yapmalısınız');
      return false;
    }

    const userId = userSession.session.user.id;

    const { error } = await supabase
      .from('tracked_stocks')
      .delete()
      .eq('stock_id', stockId)
      .eq('user_id', userId);

    if (error) {
      console.error('Error untracking stock:', error);
      toast.error('Hisse takipten çıkarılamadı');
      return false;
    }
    
    return true;
  } catch (error) {
    console.error('Error in untrackStock:', error);
    toast.error('Hisse takipten çıkarılamadı');
    return false;
  }
}
