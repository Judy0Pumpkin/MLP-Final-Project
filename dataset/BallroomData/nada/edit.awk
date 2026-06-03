/<table/		{ row = 0; col = 0; r = 0; c = 0; }
/<\/table/		{ started = 0; }
/<[tT][rR]/		{ col = 0; row++; r = 1; }
/<\/[tT][rR]/	{ r = 0; }
/<[tT][dD]/		{ col++; c = 1; }
/<\/[tT][dD]/	{ c = 0; }
/<[aA] [hH][rR][eE][fF]=/	{
					link = substr($0, 10, length($0) - 10); }
/<IMG/			{
					if (started) {
						if (index($0, "audio.gif") == 0)
							print substr($0, 58, 1), "star";
						else 
							print link;
					}
				}
/SONG TITLE/	{
					started = 1;
					print "SOURCE", link;
				}
/^[^<]/			{
					x = index($0, "<");
					if (x == 0)
						name = $0;
					else
						name = substr($0, 0, x);
					if (c && r && started && (row > 1))
						if (col < 4)
							print link, name;
						else
							print name;
				#	else if (started)
				#		print "ERROR ::", row, col, name, link;
				}
