%% Implemented by Cris Benge (cris.benge@berkeley.edu)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Class Identification %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand\fileversion{1.0}
\newcommand\filedate{2021/01/10}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{caldoc}[\filedate\space UC Berkeley, MIDS program
template for Finals and Assignments by Cris Benge (v\fileversion)]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLASS OPTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\DeclareOption{final} { \def \cal@papertype{final project}}
\DeclareOption{assignment} { \def \cal@papertype{assignment}}

\DeclareOption{colophon}  { \def \cal@colophon{true}  }
\DeclareOption{nocolophon}{ \def \cal@colophon{false} }


%\RequirePackage{silence} % :-\
%	\WarningFilter{scrbook}{Usage of package `titlesec'}
%	\WarningFilter{titlesec}{Non standard sectioning command detected}

\AtBeginDocument
{
	\@ifpackageloaded{url}
	{\urlstyle{same}}
	{\ClassWarningNoLine{caldoc}{Package 'url' missing}}
	\@ifpackageloaded{hyperref}
	{\hypersetup{%
			colorlinks=true, 
			linktocpage=true,
			pdfstartpage=1,
			pdfstartview=FitV,
			breaklinks=true,
			pageanchor=true,
			pdfpagemode=UseOutlines,
			plainpages=false, bookmarksnumbered, bookmarksopen=true, bookmarksopenlevel=1,%
			hypertexnames=true, pdfhighlight=/O,%
			urlcolor=calberkeleyblue, linkcolor=calfoundersrock, citecolor=calrosegarden}
	}
	{\ClassWarningNoLine{caldoc}{Package 'hyperref' missing}}
}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrbook}}

\ExecuteOptions{assignment}
\ExecuteOptions{nocolophon}
\ProcessOptions
\relax

\LoadClass[11pt,a4paper,fleqn]{scrbook}
\PassOptionsToPackage{
	drafting=false,    % print version information on the bottom of the pages
	tocaligned=true, % the left column of the toc will be aligned (no indentation)
	dottedtoc=true,  % page numbers in ToC flushed right
	eulerchapternumbers=true, % use AMS Euler for chapter font (otherwise Palatino)
	linedheaders=false,       % chaper headers will have line above and beneath
	floatperchapter=true,     % numbering per chapter for all floats (i.e., Figure 1.1)
	eulermath=false,  % use awesome Euler fonts for mathematical formulae (only with pdfLaTeX)
	beramono=true,    % toggle a nice monospaced font (w/ bold)
	palatino=true,    % deactivate standard font for loading another one, see the last section at the end of this file for suggestions
	style=classicthesis % classicthesis, arsclassica
}{classicthesis}
\usepackage{classicthesis}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PACKAGES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{datetime}
\usepackage{amsmath}
\RequirePackage{keyval}
\RequirePackage{etoolbox}
\RequirePackage{textcomp}
%\RequirePackage[dvipsnames, svgnames, cmyk]{xcolor}
\RequirePackage{pdfpages}
\RequirePackage{graphicx}
\graphicspath{{images/}}
\RequirePackage{tikz}
\usetikzlibrary{calc}
\RequirePackage{titling}
\usepackage{scrhack} % fix warnings when using KOMA with listings package
%\usepackage{subfig}
\usepackage{wrapfig}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LAYOUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[letterpaper, top=1in, bottom=1in,left=1in,right=1in,showframe=false]{geometry}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TITLE LAYOUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\caltitle}
{
	\begin{titlingpage}
		%\sffamily
		%\renewcommand{\and}{\vskip1mm}
		%\newcommand{\AND}{\vskip9mm}
		
		\calccentering{\unitlength}{0pt}

		\begin{center}
			\hbox{}
			\vskip5mm
	        {
	        	\Huge
				\thetitle
				\par
			}

			\ifcsempty{\cal@subtitle}
			{}
			{
				\vskip9mm
				\large
				\emph
				\cal@subtitle
			}
		
			\vskip15mm
			\begin{center}
				\Huge
				\scalebox{1.7}{\raisebox{0.15em}{\rotatebox{-70}{\int}}\kern-0.80em\rotatebox{-70}{\int}}
			\end{center}

			\vskip15mm
			{
				\large
				\begin{table}[!ht]
					\centering
					\begin{minipage}[t]{0.24\textwidth}
						\centering
						\begin{tabular}[t]{c}
							{\Large\href{https://www.linkedin.com/in/siduojiang/}{Siduo Jiang}} \\
							\textcolor{calpacific}{\scriptsize{\texttt{siduojiang@berkeley.edu}}}\\
						\end{tabular}
					\end{minipage} \hfill
					\begin{minipage}[t]{0.24\textwidth}
						\centering
						\begin{tabular}[t]{c}
							{\Large\href{https://cbenge509.github.io/}{Cristopher Benge}} \\
							\textcolor{calpacific}{\scriptsize{\texttt{cris.benge@berkeley.edu}}}\\
						\end{tabular}
					\end{minipage} \hfill
					\begin{minipage}[t]{0.24\textwidth}
						\centering
						\begin{tabular}[t]{c}
							{\Large\href{https://jackson.yale.edu/person/casey-king/}{Casey King}} \\
							\textcolor{calpacific}{\scriptsize{\texttt{caseyking@berkeley.edu}}}\\
						\end{tabular}
					\end{minipage} \hfill
					\begin{minipage}[t]{0.24\textwidth}
						\centering
						\begin{tabular}[t]{c}
							{\Large\href{http://seekinginference.com/}{Andrew Fogarty}} \\
							\textcolor{calpacific}{\scriptsize{\texttt{apfogarty@berkeley.edu}}}\\
						\end{tabular}
					\end{minipage} \hfill
				\end{table}	

			}
		
			\vskip10mm
			{
				\includegraphics[width = 60mm]{images/UCBerkeley.png}%
			}

			\vskip10mm
			{
				\large
				\textit
				{
					Submitted in partial fulfillment of the requirements \\
					for the \cal@papertype\space of MIDS W210 Capstone \\
					to the \\
					Faculty of Graduate Studies \\
					of the School of Information \\
					at the University of California, Berkeley
				}
			}

		
		\end{center}

		%\null
		%\clearpage
		\ifdefstring{\cal@colophon}{true}{\cal@print@colophon}{}
				
	\end{titlingpage}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% COLOPHON %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\cal@print@colophon}
{
	\newpage
	\thispagestyle{empty}
	\thispagestyle{firstpage} % Apply the page style for the first page (no headers and footers)
	%	\vspace*{\stretch{3}}
	\vspace*{\fill}
	\begin{flushleft}
		\begin{minipage}[c][30mm]{20mm}
			\includegraphics[width = 15mm]{github.png}
		\end{minipage}%
		\begin{minipage}[c]{\textwidth - 20mm}
			\Large
			\texttt{\textcolor{calsouthhall}{Project GitHub Repository}: \href{\cal@repourl}{\cal@reponame}}
			%\texttt{\textcolor{calsouthhall}{Project GitHub Repository}: \href{https://github.com/cbenge509/BERTVision}{BERTVision}}
		\end{minipage}%
		\vskip4\baselineskip
		\normalsize
		\textbf{\sffamily\textcopyright\space\theauthor\space\textcolor{calrosegarden}{:}\space\monthname, \the\year}
		\vskip2\baselineskip
		Publication submitted to the graduate studies faculty of the University of California, Berkeley for evaluation and consideration: \space
		\textcolor{calpacific}{\textit{\cal@faculty}}
		\par
		\vskip\baselineskip
		\small
		This document was typeset using \LaTeX, with a mixture of \texttt{classicthesis} developed by André Miede.  The bibliography was processed by Biblatex. 
		Robert Slimbach's Minion Pro acts as both the text and display typeface. Sans-serif text is typeset in Slimbach and Carol Twombly's
		\fontfamily{Myriad Pro}{Myriad Pro}; monospaced text uses Jim Lyle's \texttt{Bitstream Vera Mono}.
		\vskip2.5\baselineskip
		\small
		\textcopyright\space \emph{BERT VISION : Improving span annotation and classification task performance using parameter-efficient model architectures
			trained on BERT's hidden state activations.}
		\vskip1\baselineskip
		All rights reserved. No part of this publication may be reproduced or transmitted, in any form or by any means, without permission from the authors.
	\end{flushleft}
	\null
	\newpage
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% UTILITIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\definecolor{callink} {HTML}{0B5A9D}

\newcommand{\department}[1]{ \def \cal@dept{#1} }
\newcommand{\githubrepo}[2]{ \def \cal@reponame{#1} \def \cal@repourl{#2} }
\newcommand{\faculty}[1]{ \def \cal@faculty{#1} }
\renewcommand{\subtitle}[1]{ \def \cal@subtitle{#1} }

\subtitle{}
\department{}
\faculty{}
\githubrepo{}{}

\newcommand{\fakesection}[1]{%
	\par\refstepcounter{section}% Increase section counter
	\sectionmark{#1}% Add section mark (header)
	\addcontentsline{toc}{section}{\protect\numberline{\thesection}#1}% Add section to ToC
	% Add more content here, if needed.
}

\newcommand{\fakesubsection}[1]{%
	\par\refstepcounter{subsection}% Increase subsection counter
	\subsectionmark{#1}% Add subsection mark (header)
	\addcontentsline{toc}{subsection}{\protect\numberline{\thesubsection}#1}% Add subsection to ToC
	% Add more content here, if needed.
}

\def\doubleline{
	
	\noindent\rule{\linewidth}{0.3pt}%

	\vspace{-0.95em}%
	\noindent\rule{\linewidth}{0.3pt}%
	\vspace{8pt}%	
}
