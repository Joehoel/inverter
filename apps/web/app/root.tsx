import "@mantine/core/styles.css";
import "@mantine/notifications/styles.css";
import "@mantine/nprogress/styles.css";
import "@mantine/dropzone/styles.css";

import { ColorSchemeScript, MantineProvider, AppShell } from "@mantine/core";
import { NavigationProgress, nprogress } from "@mantine/nprogress";
import { cssBundleHref } from "@remix-run/css-bundle";
import type { LinksFunction } from "@remix-run/node";
import {
  Links,
  LiveReload,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
  useFetchers,
  useNavigation,
  useRouteError,
} from "@remix-run/react";
import { useEffect } from "react";
import { NotFound } from "./components/not-found";
import { Navbar } from "./components/navbar";

export const links: LinksFunction = () => [
  ...(cssBundleHref ? [{ rel: "stylesheet", href: cssBundleHref }] : []),
];

export default function App() {
  const navigation = useNavigation();
  const fetchers = useFetchers();
  useEffect(() => {
    const fetchersIdle = fetchers.every((f) => f.state === "idle");
    if (navigation.state === "idle" && fetchersIdle) {
      nprogress.complete();
    } else {
      nprogress.start();
    }

    return () => {
      nprogress.cleanup();
    };
  }, [navigation.state, fetchers]);

  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
        <ColorSchemeScript />
      </head>
      <body>
        <MantineProvider>
          <NavigationProgress />
          <AppShell>
            <AppShell.Navbar>
              <Navbar />
            </AppShell.Navbar>
            <AppShell.Main>
              <Outlet />
            </AppShell.Main>
          </AppShell>
          <ScrollRestoration />
          <Scripts />
          <LiveReload />
        </MantineProvider>
      </body>
    </html>
  );
}

export function ErrorBoundary() {
  const error = useRouteError();
  console.error(error);
  return (
    <html lang="en">
      <head>
        <title>Oh no!</title>
        <Meta />
        <Links />
        <ColorSchemeScript />
      </head>
      <body>
        <MantineProvider>
          <NavigationProgress />
          <NotFound />
          <Scripts />
        </MantineProvider>
      </body>
    </html>
  );
}
