    """Fetch NSE index data from Wikipedia"""
        try:
            # Fetch tables from Wikipedia page
            url = 'https://en.wikipedia.org/wiki/NSE_Indices'
            tables = pd.read_html(url)
            
            # The main indices table is typically one of the first tables
            # We'll need to process multiple tables to get different types of indices
            indices = {}
            
            # Process Broad Market Indices
            broad_market = tables[1]  # Adjust index based on actual table position
            for _, row in broad_market.iterrows():
                index_name = row['Index Name']
                if pd.notna(index_name):  # Check if index name is not NaN
                    # Convert index name to Yahoo Finance symbol format
                    symbol = self.get_yahoo_finance_symbol(index_name)
                    if symbol:
                        indices[index_name] = symbol
            
            # Process Sectoral Indices
            sectoral = tables[2]  # Adjust index based on actual table position
            for _, row in sectoral.iterrows():
                index_name = row['Index Name']
                if pd.notna(index_name):
                    symbol = self.get_yahoo_finance_symbol(index_name)
                    if symbol:
                        indices[index_name] = symbol
            
            # If no indices were found, use default indices as fallback
            if not indices:
                logger.warning("No indices found from Wikipedia, using default indices")
                indices = {
                    'NIFTY 50': '^NSEI',
                    'SENSEX': '^BSESN',
                    'NIFTY BANK': '^NSEBANK',
                    'NIFTY IT': '^CNXIT',
                    'NIFTY AUTO': '^0P0001PQB7'
                }
            
            logger.info(f"Successfully fetched {len(indices)} indices")
            return indices
        
        except Exception as e:
            logger.error(f"Error fetching index data from Wikipedia: {e}")
            # Return default indices as fallback
            return {
                'NIFTY 50': '^NSEI',
                'SENSEX': '^BSESN',
                'NIFTY BANK': '^NSEBANK',
                'NIFTY IT': '^CNXIT',
                'NIFTY AUTO': '^0P0001PQB7'
            }

    def get_yahoo_finance_symbol(self, index_name):
        """Convert index name to Yahoo Finance symbol"""
        # Mapping of index names to Yahoo Finance symbols
        symbol_mapping = {
            'NIFTY 50': '^NSEI',
            'NIFTY NEXT 50': '^CNXJUNIOR',
            'NIFTY 100': '^CNX100',
            'NIFTY 200': '^CNX200',
            'NIFTY 500': '^CRSLDX',
            'NIFTY BANK': '^NSEBANK',
            'NIFTY IT': '^CNXIT',
            'NIFTY AUTO': '^0P0001PQB7',
            'NIFTY FINANCIAL SERVICES': '^CNXFINANCE',
            'NIFTY FMCG': '^CNXFMCG',
            'NIFTY MEDIA': '^CNXMEDIA',
            'NIFTY METAL': '^CNXMETAL',
            'NIFTY PHARMA': '^CNXPHARMA',
            'NIFTY PSU BANK': '^CNXPSUBANK',
            'NIFTY PRIVATE BANK': '^NIFTYPRBANK',
            'NIFTY REALTY': '^CNXREALTY',
            # Add more mappings as needed
        }
        
        # Try to find exact match
        if index_name in symbol_mapping:
            return symbol_mapping[index_name]
        
        # Try to find case-insensitive match
        index_name_upper = index_name.upper()
        for known_name, symbol in symbol_mapping.items():
            if known_name.upper() == index_name_upper:
                return symbol
        
        logger.warning(f"No Yahoo Finance symbol found for index: {index_name}")
        return None

    def validate_indices(self):
        """Validate that we can fetch data for each index"""
        valid_indices = {}
        for index_name, symbol in self.indices.items():
            if self.is_ticker_valid(symbol):
                valid_indices[index_name] = symbol
            else:
                logger.warning(f"Removing invalid index: {index_name} ({symbol})")
        
        self.indices = valid_indices
        logger.info(f"Validated {len(valid_indices)} indices")