"""
网络搜索模块
同时搜索Brave和DuckDuckGo，处理超时和失败
"""
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup


class WebSearchManager:
    """双引擎网络搜索管理器"""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.timeout = 15

    def search(self, query: str) -> Dict:
        """
        同时向Brave和DuckDuckGo发起搜索

        Returns:
            Dict: {'success': bool, 'results': list, 'errors': list}
        """
        results = []
        errors = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self._search_duckduckgo, query): 'DuckDuckGo',
                executor.submit(self._search_brave, query): 'Brave',
            }

            for future in as_completed(futures):
                source = futures[future]
                try:
                    data = future.result(timeout=self.timeout + 5)
                    if data:
                        results.extend(data)
                except Exception as e:
                    errors.append(f"{source}: {str(e)}")
                    print(f"⚠️ {source}搜索失败: {str(e)}")

        # 去重（按标题）
        seen = set()
        unique_results = []
        for r in results:
            title = r.get('title', '')
            if title and title not in seen:
                seen.add(title)
                unique_results.append(r)
            elif not title:
                unique_results.append(r)

        return {
            'success': len(unique_results) > 0,
            'results': unique_results[:self.max_results * 2],
            'errors': errors
        }

    def _search_duckduckgo(self, query: str) -> List[Dict]:
        """DuckDuckGo搜索"""
        try:
            from ddgs import DDGS
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=self.max_results, region='cn-zh'))

            formatted = []
            for r in results:
                formatted.append({
                    'title': r.get('title', ''),
                    'link': r.get('href', ''),
                    'snippet': r.get('body', '')[:300],
                    'source': 'DuckDuckGo'
                })
            return formatted
        except ImportError:
            try:
                from duckduckgo_search import DDGS
                ddgs = DDGS()
                results = list(ddgs.text(query, max_results=self.max_results))
                return [{
                    'title': r.get('title', ''),
                    'link': r.get('link', r.get('href', '')),
                    'snippet': r.get('snippet', r.get('body', ''))[:300],
                    'source': 'DuckDuckGo'
                } for r in results]
            except Exception as e:
                print(f"⚠️ DuckDuckGo搜索失败: {str(e)}")
                return []
        except Exception as e:
            print(f"⚠️ DuckDuckGo搜索失败: {str(e)}")
            return []

    def _search_brave(self, query: str) -> List[Dict]:
        """Brave搜索（无需API Key的网页抓取方式）"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            }

            resp = requests.get(
                'https://search.brave.com/search',
                params={'q': query, 'source': 'web'},
                headers=headers,
                timeout=self.timeout
            )

            if resp.status_code != 200:
                return []

            soup = BeautifulSoup(resp.text, 'html.parser')
            results = []

            # 解析Brave搜索结果
            for snippet in soup.select('.snippet, .result, [data-type="web"]')[:self.max_results]:
                title_el = snippet.select_one('.snippet-title, .title, h3')
                link_el = snippet.select_one('a[href]')
                desc_el = snippet.select_one('.snippet-description, .description, p')

                title = title_el.get_text(strip=True) if title_el else ''
                link = link_el.get('href', '') if link_el else ''
                description = desc_el.get_text(strip=True) if desc_el else ''

                if title and description:
                    results.append({
                        'title': title,
                        'link': link,
                        'snippet': description[:300],
                        'source': 'Brave'
                    })

            return results

        except requests.exceptions.Timeout:
            print("⚠️ Brave搜索超时")
            return []
        except Exception as e:
            print(f"⚠️ Brave搜索失败: {str(e)}")
            return []

    def fetch_page_content(self, url: str, max_length: int = 1000) -> str:
        """获取网页内容"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            resp = requests.get(url, headers=headers, timeout=10)
            resp.encoding = 'utf-8'

            soup = BeautifulSoup(resp.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            return text[:max_length]
        except:
            return ""