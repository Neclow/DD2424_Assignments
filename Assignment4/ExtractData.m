function book_data = ExtractData(book_fname)
fid = fopen(book_fname, 'r');
book_data = fscanf(fid, '%c');
fclose(fid);
end