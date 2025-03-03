
import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import Header from '@/components/Header';
import { StockNewsMapping } from '@/utils/types';

const StockNewsMapping = () => {
  const [mappingData, setMappingData] = useState<StockNewsMapping | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMappingData = async () => {
      try {
        setIsLoading(true);
        const response = await fetch('/stock_news_mapping.json');
        
        if (!response.ok) {
          throw new Error(`Failed to fetch mapping data: ${response.status}`);
        }
        
        const data = await response.json();
        setMappingData(data);
      } catch (err) {
        console.error('Error fetching stock news mapping:', err);
        setError('Failed to load stock news mapping data');
      } finally {
        setIsLoading(false);
      }
    };

    fetchMappingData();
  }, []);

  return (
    <div className="container mx-auto px-4">
      <Header onSettingsClick={() => {}} />
      
      <div className="my-8 space-y-6">
        <h1 className="text-2xl font-bold">Hisse Haberleri Eşleştirme</h1>
        
        {isLoading && (
          <div className="flex items-center justify-center py-10">
            <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
          </div>
        )}
        
        {error && (
          <div className="bg-red-50 border border-red-300 text-red-800 p-4 rounded-md">
            <p>{error}</p>
          </div>
        )}
        
        {mappingData && (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Güncelleme Bilgisi</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Güncelleme Zamanı:</p>
                    <p className="text-lg font-medium">{mappingData.timestamp}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Güncellenmiş:</p>
                    <p className="text-lg font-medium">{mappingData.updated ? 'Evet' : 'Hayır'}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Hisse Haberleri</CardTitle>
              </CardHeader>
              <CardContent>
                {Object.keys(mappingData.stock_news).length === 0 ? (
                  <p className="text-muted-foreground">Henüz hisse haberi bulunmamaktadır.</p>
                ) : (
                  <div className="space-y-6">
                    {Object.entries(mappingData.stock_news).map(([symbol, urls]) => (
                      <div key={symbol} className="border-b pb-4 last:border-0 last:pb-0">
                        <h3 className="text-lg font-semibold mb-2">{symbol}</h3>
                        <ul className="space-y-2">
                          {urls.map((url, index) => (
                            <li key={index} className="text-sm">
                              <a 
                                href={url} 
                                target="_blank" 
                                rel="noopener noreferrer"
                                className="text-blue-600 hover:underline break-all"
                              >
                                {url}
                              </a>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
};

export default StockNewsMapping;
