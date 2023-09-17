using Franklin
using Bibliography
using BibInternal
using DataStructures: OrderedDict

abstract type Reference end

struct URLReference <: Reference
    url::String
end

struct BibReference <: Reference
    label::String
end

function getrefcounts()
    refcounts = Franklin.locvar("refcounts")
    (refcounts != nothing) && return refcounts
    refcounts = OrderedDict{Reference, Int}()
    Franklin.set_var!(Franklin.LOCAL_VARS, "refcounts", refcounts)
    refcounts
end

function refcount(ref)
    refcounts = getrefcounts()
    isfirst = !haskey(refcounts, ref)
    isfirst && (refcounts[ref] = length(refcounts) + 1)
    refcounts[ref], isfirst
end

function getbibliography()
    bib = Franklin.locvar("bibliography")
    (bib != nothing) && return bib
    curpath = joinpath(Franklin.path(:folder), locvar("fd_rpath"))
    files = collect(filter_files(endswith(".bib"), dirname(curpath)))
    @assert !isempty(files) "no .bib file found for citation!"
    @assert length(files) == 1 "more than 1 .bib file in folder"
    bib = import_bibtex(first(files))
    Franklin.set_var!(Franklin.LOCAL_VARS, "bibliography", bib)
    bib
end

maybe(x, pre=" ", post="") = isempty(x) ? "" : "$pre$x$post"
emph(x) = "<em>$x</em>"
hrefto(href, content) = """<a href="$href" target="_blank" rel="noopener">$content</a>"""
href(href, content) = """<a href="$href">$content</a>"""
bibentry(x::BibReference) = get(getbibliography(), x.label, "MISSING REFERENCE")

format(x::URLReference) = hrefto(x.url, x.url)
format(x::BibReference) = format(bibentry(x))
format(x::Vector{BibInternal.Name}) = join(map(format, x), ", ", ", & ")
format(x::BibInternal.Name) = "$(x.last), $(first(x.first))."
format(x::BibInternal.Date) = x.year
format(x::String) = x

function format(x::BibInternal.Entry)
    str = if(x.type == "inproceedings")
        # TODO: pages and link
        """$(format(x.authors)) ($(format(x.date))). $(x.title). \
           In $(emph(x.booktitle))$(maybe(x.in.pages, " (pp. ", ")"))."""
    elseif(x.type == "article")
        """$(format(x.authors)) ($(format(x.date))). $(x.title). \
           In $(emph(x.in.journal))$(maybe(x.in.volume))$(maybe(x.in.pages, " (pp. ", ")"))."""
    else
        """Unsupported bibtex entry"""
    end
    str = replace(str, r"\-{2,}" => "-") # double/triple dash to single dash
    replace(str, r"[\n\r ]+" => " ") # trim newlines
end

function hfun_citeurl(params)
    @assert length(params) == 1 "citeurl only takes 1 parameter!"
    ref = URLReference(first(params))
    count, isfirst = refcount(ref)
    backref = isfirst ? """id="ref-$count" """ : ""
    """<sup class="sup-ref"><a $(maybe(backref))href="#bib-$count">$count</a></sup>"""
end

function hfun_cite(params)
    @assert length(params) == 1 "cite only takes 1 parameter!"
    ref = BibReference(first(params))
    count, isfirst = refcount(ref)
    backref = isfirst ? "id=\"ref-$count\"" : ""
    """<sup class="sup-ref"><a $(maybe(backref))href="#bib-$count">$count</a></sup>"""
end

function hfun_citep(params)
    @assert length(params) == 1 "citep only takes 1 parameter!"
    ref = BibReference(first(params))
    count, isfirst = refcount(ref)
    entry = bibentry(ref)
    backref = isfirst ? """id="ref-$count" """ : ""
    """<a $(maybe(backref))href="#bib-$count">($(format(entry.authors)), $(format(entry.date)))</a>"""
end

function hfun_bibliography()
    bib, refcounts = getbibliography(), getrefcounts()
    io = IOBuffer()
    write(io, """<h2 class="bib-heading">References</h2>""")
    write(io, """<ol class="bib">""")
    for (ref, count) in refcounts
        write(io, """<li class="bib-entry" id="bib-$(count)">""")
        write(io, format(ref))
        write(io, """ <a href="#ref-$count">↩︎</a>""")
        write(io, """</li>""")
    end
    write(io, """</ol>""")
    String(take!(io))
end

function hfun_hascitations()
    return locvar("ref_counts") != nothing
end
