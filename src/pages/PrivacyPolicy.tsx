
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AuthLayout } from '@/components/auth/AuthLayout';
import { Button } from '@/components/ui/button';
import { useNavigate } from 'react-router-dom';
import { ChevronLeft } from 'lucide-react';

export default function PrivacyPolicy() {
  const navigate = useNavigate();

  const handleGoBack = () => {
    navigate(-1);
  };

  return (
    <AuthLayout>
      <div className="w-full max-w-4xl mx-auto">
        <Button 
          variant="ghost" 
          onClick={handleGoBack} 
          className="mb-4"
        >
          <ChevronLeft className="h-4 w-4 mr-2" />
          Geri Dön
        </Button>

        <Card className="w-full">
          <CardHeader>
            <CardTitle className="text-2xl font-bold text-center">Gizlilik Politikası</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-sm md:text-base">
            <div className="space-y-4">
              <h2 className="text-xl font-semibold">Genel Bakış</h2>
              <p>
                Stock News Signal uygulaması olarak, gizliliğinize saygı duyuyoruz. Bu gizlilik politikası, hizmetlerimizi 
                kullanırken hangi bilgileri topladığımızı ve bunları nasıl kullandığımızı açıklamaktadır.
              </p>
              
              <h2 className="text-xl font-semibold">Toplanan Bilgiler</h2>
              <p>
                <strong>Telefon Numarası:</strong> Sadece WhatsApp bildirimleri göndermek amacıyla telefon numaranızı topluyoruz. 
                Bu, takip ettiğiniz hisseler hakkında güncel haberler alabilmeniz için kullanılır.
              </p>
              
              <h2 className="text-xl font-semibold">Bilgilerin Kullanımı</h2>
              <p>
                Topladığımız telefon numarası bilgisi yalnızca aşağıdaki amaçlar için kullanılır:
              </p>
              <ul className="list-disc pl-6 space-y-2">
                <li>Size takip ettiğiniz hisseler hakkında WhatsApp üzerinden bildirimler göndermek</li>
                <li>Hesabınızın güvenliğini sağlamak ve doğrulamak</li>
              </ul>
              
              <h2 className="text-xl font-semibold">Veri Paylaşımı</h2>
              <p>
                Telefon numaranız, WhatsApp bildirimleri gönderebilmemiz için WhatsApp Business API ile paylaşılır. 
                Bunun dışında, bilgilerinizi üçüncü taraflarla paylaşmıyoruz veya satmıyoruz.
              </p>
              
              <h2 className="text-xl font-semibold">Veri Güvenliği</h2>
              <p>
                Bilgilerinizi korumak için uygun güvenlik önlemleri alıyoruz. Telefon numaranız güvenli bir şekilde 
                saklanır ve yalnızca gerekli bildirim amaçları için kullanılır.
              </p>
              
              <h2 className="text-xl font-semibold">Haklarınız</h2>
              <p>
                Verileriniz hakkında bilgi alma, silme veya düzeltme talebinde bulunma hakkınız vardır. 
                Herhangi bir sorunuz veya endişeniz varsa, lütfen bizimle iletişime geçin.
              </p>
              
              <h2 className="text-xl font-semibold">Değişiklikler</h2>
              <p>
                Bu gizlilik politikasını zaman zaman güncelleyebiliriz. Herhangi bir değişiklik yapılması durumunda, 
                güncellenmiş politikayı burada yayınlayacağız.
              </p>
              
              <h2 className="text-xl font-semibold">İletişim</h2>
              <p>
                Bu gizlilik politikası hakkında sorularınız varsa, lütfen bizimle iletişime geçin.
              </p>
              
              <p className="text-sm text-muted-foreground pt-4">
                Son güncelleme: {new Date().toLocaleDateString('tr-TR')}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </AuthLayout>
  );
}
